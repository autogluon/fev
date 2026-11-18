"""Present each provider with the observed cohort at a common cutoff."""

import json
from copy import copy
from dataclasses import replace
from hashlib import sha256

import datasets
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from .types import SeriesPrefix


def predictions(task, values):
    columns = ["predictions", *map(str, task.quantile_levels)]
    output = {}
    for j, target in enumerate(task.target_columns):
        fields = {
            name: pa.FixedSizeListArray.from_arrays(
                pa.array(np.ascontiguousarray(values[:, :, j, k]).ravel()), task.horizon
            )
            for k, name in enumerate(columns)
        }
        output[target] = datasets.Dataset(pa.table(fields)).cast(task.predictions_schema)
    return datasets.DatasetDict(output)


def fast(values, clock=False):
    dtype = values.type
    return (not values.null_count and
            (pa.types.is_floating(dtype) or pa.types.is_signed_integer(dtype) or pa.types.is_boolean(dtype) or
             (clock and pa.types.is_timestamp(dtype) and dtype.tz is None)))


def convert(row, clock=False):
    if not row.is_valid or not fast(row.values, clock):
        return row.as_py()
    values = row.values
    array = values.to_numpy(zero_copy_only=False)
    if pa.types.is_floating(values.type):
        return array.astype(np.float64, copy=False)
    if pa.types.is_signed_integer(values.type):
        return array.astype(np.int64, copy=False)
    return array


def joined(a, b):
    if a.is_valid and b.is_valid and a.values.type == b.values.type and fast(a.values) and fast(b.values):
        return np.concatenate((convert(a), convert(b)))
    return a.as_py() + b.as_py()


def typed_panel(task, past, future):
    old, issued = past.data.table, future.data.table
    fields = {k: old.column(k) for k in (*task.target_columns, *task.past_dynamic_columns,
                                         *task.known_dynamic_columns, task.timestamp_column)}
    plans = {k: issued.column(k) for k in (*task.known_dynamic_columns, task.timestamp_column)}
    static = {k: old.column(k).to_pylist() for k in task.static_columns}
    return tuple(SeriesPrefix(
        str(uid), [convert(fields[k][i]) for k in task.target_columns],
        convert(fields[task.timestamp_column][i], clock=True), convert(plans[task.timestamp_column][i], clock=True),
        past={k: convert(fields[k][i]) for k in task.past_dynamic_columns},
        known={k: joined(fields[k][i], plans[k][i]) for k in task.known_dynamic_columns},
        static={k: static[k][i] for k in static},
    ) for i, uid in enumerate(old.column(task.id_column).to_pylist()))



def native_case(task, window, expected):
    past, future = window.get_input_data()
    shared = {task.id_column, task.timestamp_column, *task.known_dynamic_columns, *task.static_columns}
    if (set(future.column_names) != shared
            or set(past.column_names) != shared | set(task.target_columns) | set(task.past_dynamic_columns)):
        raise ValueError(f"{task.task_name} prediction fields differ from their availability roles.")
    ids = list(past[task.id_column])
    if ids != list(future[task.id_column]) or len(set(ids)) != len(ids):
        raise ValueError("Native recipient order differs between the two input roles.")
    column = past.data.table.column(task.timestamp_column)
    ends = np.asarray([row.values[-1].as_py() for row in column], dtype="datetime64[ns]")
    mapped = pc.list_flatten(future.data.table.column(task.timestamp_column)).to_numpy(zero_copy_only=False)
    dates = mapped.astype("datetime64[ns]").reshape(len(ids), task.horizon)
    if np.isnat(dates).any() or np.isnat(ends).any() or (dates[:, 0] <= ends).any():
        raise ValueError("A forecast row does not follow its own historical cutoff.")
    meta = {"origin": expected["origin"], "series": len(ids), "horizon": task.horizon,
            "target_columns": task.target_columns, "cutoff_spec": window.cutoff,
            "membership_sha256": sha256(json.dumps(ids, ensure_ascii=False).encode()).hexdigest(),
            "past_end_min_max": [np.datetime_as_string(t, unit="s") for t in (ends.min(), ends.max())]}
    if any(expected[key] != value for key, value in meta.items()):
        raise ValueError(f"{task.task_name}/{meta['origin']} native case geometry changed.")
    meta.update(issue_at_ns=int(ends.min().astype(np.int64)),
                available_at_ns=int(dates.max().astype(np.int64)))
    return (window, past, future, ids, dates, meta), ends


def issued_view(window, past, future):
    """Expose only original typed inputs to a provider-specific adapter."""
    issued = replace(window, full_dataset=None)
    issued.get_input_data = lambda: (past, future)
    return issued


def predict_at_cutoffs(predict, task, case, ends):
    window, past, future, ids, dates, meta = case
    model_task = copy(task)
    model_task._freq = task.freq
    model_task._full_dataset = None
    cutoffs = np.unique(ends)
    shape = (*dates.shape, len(task.target_columns), len(task.quantile_levels) + 1)
    if len(cutoffs) == 1:
        view = issued_view(window, past, future)
        return predict(model_task, (view, past, future, ids, dates, meta))

    values = np.empty(shape)
    for cutoff in cutoffs:
        rows = np.flatnonzero(ends == cutoff)
        # Some model adapters read the Arrow table directly.
        previous = past.select(rows).flatten_indices()
        ahead = future.select(rows).flatten_indices()
        keys = [ids[i] for i in rows]
        view = issued_view(window, previous, ahead)
        clock = np.datetime_as_string(cutoff, unit="s")
        info = meta | {"issue_at_ns": int(cutoff.astype(np.int64)), "series": len(keys),
                       "past_end_min_max": [clock, clock]}
        block = predict(model_task, (view, previous, ahead, keys, dates[rows], info))
        if block.shape != (len(rows), *shape[1:]):
            raise ValueError("The provider changed the size of an issuance cohort.")
        values[rows] = block
    return values



class EarlierInputs:
    """Resolve earlier eligible recipients and roles on the observed clock."""

    def __init__(self, task, first_case, official_ends):
        self.task = task
        self.template, self.history, _, self.ids, _, _ = first_case
        self.official_ends = official_ends
        self.minimum = self.template.min_context_length
        self.horizon = self.template.horizon
        column = self.history.data.table.column(task.timestamp_column).combine_chunks()
        offsets = column.offsets.to_numpy(zero_copy_only=False).astype(np.int64)
        self.offsets = offsets - offsets[0]
        values = pc.list_flatten(column).to_numpy(zero_copy_only=False)
        clocks = values.astype("datetime64[ns]")
        self.clocks = clocks.astype(np.int64)
        invalid = np.flatnonzero(clocks[1:] <= clocks[:-1])
        if (column.null_count or np.isnat(clocks).any()
                or not np.isin(invalid, self.offsets[1:-1] - 1, assume_unique=True).all()):
            raise ValueError("Earlier inputs require ordered nonempty series clocks.")
        lengths = np.diff(self.offsets)
        if (lengths < self.minimum).any() or not np.array_equal(clocks[self.offsets[1:] - 1], official_ends):
            raise ValueError("Original series do not match the prefix cutoff.")
        self.grid = np.unique(self.clocks)
        starts = np.searchsorted(self.grid, self.clocks[self.offsets[:-1]])
        ends = np.searchsorted(self.grid, self.clocks[self.offsets[1:] - 1])
        self.contiguous = ends - starts + 1 == lengths
        has_room = lengths > self.minimum + self.horizon
        active = np.flatnonzero(has_room)
        self.first = np.full(len(lengths), len(self.grid))
        self.last = np.full(len(lengths), -1)
        self.first[active] = np.searchsorted(self.grid, self.clocks[self.offsets[active] + self.minimum - 1])
        self.last[active] = np.searchsorted(self.grid, self.clocks[self.offsets[active + 1] - self.horizon - 2])
        intervals = active[self.contiguous[active]]
        boundaries = np.zeros(len(self.grid) + 1, dtype=np.int64)
        np.add.at(boundaries, self.first[intervals], 1)
        np.add.at(boundaries, self.last[intervals] + 1, -1)
        counts = boundaries.cumsum()[:-1]
        self.ragged = active[~self.contiguous[active]]
        for i in self.ragged:
            np.add.at(counts, np.searchsorted(self.grid, self.item_clock(i)), 1)
        self.active_grid = np.flatnonzero(counts > 0)
        self.counts = counts[self.active_grid]
        if self.counts.sum() != int(np.maximum(lengths - self.minimum - self.horizon, 0).sum()):
            raise ValueError("Clock allocation omitted or repeated an earlier item origin.")

    def item_clock(self, i):
        return self.clocks[self.offsets[i] + self.minimum - 1 : self.offsets[i + 1] - self.horizon - 1]

    def case(self, index):
        grid_row = self.active_grid[index]
        stamp = int(self.grid[grid_row])
        present = (self.first <= grid_row) & (grid_row <= self.last)
        for i in self.ragged[present[self.ragged]]:
            eligible_dates = self.item_clock(i)
            j = np.searchsorted(eligible_dates, stamp)
            present[i] = j < len(eligible_dates) and eligible_dates[j] == stamp
        order = np.flatnonzero(present)
        ids = [self.ids[i] for i in order]
        if len(ids) != self.counts[index]:
            raise ValueError("Cutoff recipient membership differs from schedule.")
        tick = np.datetime64(stamp, "ns")
        cutoff = np.datetime_as_string(tick, unit="ms")
        window = replace(self.template, full_dataset=self.history.select(order), cutoff=cutoff,
                         max_context_length=None)
        second = np.datetime_as_string(tick, unit="s")
        expected = {"origin": stamp, "cutoff_spec": cutoff, "series": len(ids),
                    "membership_sha256": sha256(json.dumps(ids, ensure_ascii=False).encode()).hexdigest(),
                    "past_end_min_max": [second, second],
                    "horizon": self.horizon, "target_columns": self.task.target_columns}
        case, ends = native_case(self.task, window, expected)
        if (case[3] != ids or not (ends == tick).all()
                or (case[4][:, -1] >= self.official_ends[order]).any()):
            raise ValueError("A historical view would cross its own official cutoff.")
        return case, ends, order

    def describe(self):
        return {
            "origins": len(self.active_grid), "item_origins": int(self.counts.sum()),
            "target_lead_rows": int(self.counts.sum()) * self.horizon * len(self.task.target_columns),
            "largest_cohort": int(self.counts.max(initial=0)),
            "ragged_items": len(self.ragged),
            "first_cutoff_ns": int(self.grid[self.active_grid[0]]) if len(self.active_grid) else None,
            "last_cutoff_ns": int(self.grid[self.active_grid[-1]]) if len(self.active_grid) else None,
        }

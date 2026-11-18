"""Calibration may use only the native first window's visible history."""

from datetime import datetime, timedelta

import datasets
import numpy as np
import pytest
from _dwabi_fev.cases import EarlierInputs
from _dwabi_fev.cutoff import issue
from _dwabi_fev.native_calendar import NativeEarlierInputs, build_calendars, capture_native, earlier_episodes
from dwabi import CutoffDecision
from model import DWabiFEVModel
from prepare_cutoff import fit

from fev import Task


def native_task(
    tmp_path,
    name,
    *,
    cap=18,
    minimum=3,
    hidden=0.0,
    visible=20.0,
    extra_prefix=0,
    staggered=False,
    multitarget=False,
    shift=0,
):
    ids = ["a", "b"] if staggered else ["a"]
    clocks, targets, second, drivers, sensors = [], [], [], [], []
    for index, _ in enumerate(ids):
        offsets = range(-extra_prefix, 82)
        dates = [datetime(2020, 1, 1) + timedelta(days=day + shift + 7 * index) for day in offsets]
        values = [
            float((hidden if day < (80 - cap if cap is not None else 62) else visible) + day % 2) for day in offsets
        ]
        clocks.append(dates)
        targets.append(values)
        second.append([2 * value + 1 for value in values])
        drivers.append([float(hidden if day < (80 - cap if cap is not None else 62) else 3) for day in offsets])
        sensors.append([float(hidden if day < (80 - cap if cap is not None else 62) else 4) for day in offsets])
    data = {
        "id": ids,
        "timestamp": clocks,
        "target": targets,
        "driver": drivers,
        "sensor": sensors,
        "static": list(range(len(ids))),
    }
    if multitarget:
        data["target_2"] = second
    path = tmp_path / f"{name}-{hidden}-{visible}-{extra_prefix}-{cap}-{shift}.parquet"
    datasets.Dataset.from_dict(data).to_parquet(str(path))
    return Task(
        dataset_path=str(path),
        task_name=name,
        horizon=2,
        num_windows=1,
        min_context_length=minimum,
        max_context_length=cap,
        seasonality=1,
        target=["target", "target_2"] if multitarget else "target",
        known_dynamic_columns=["driver"],
        past_dynamic_columns=["sensor"],
        static_columns=["static"],
        eval_metric="SQL",
        extra_metrics=["MASE", {"name": "WAPE", "epsilon": 1}, {"name": "WQL", "epsilon": 1}],
        quantile_levels=[i / 10 for i in range(1, 10)],
    )


def fitted(task, sibling=None):
    tasks = [task] if sibling is None else [sibling, task]
    schedule, local = build_calendars(tasks, dict.fromkeys((t.task_name for t in tasks), "source"))
    model = DWabiFEVModel(device="cpu")
    received = []
    for index, name in enumerate(model.provider_pins):

        def forecast(current_task, case, value=10.0 * index):
            received.append((case[1].data.table, case[2].data.table))
            return np.full((*case[4].shape, len(current_task.target_columns), 10), value)

        model.providers[name] = forecast
    name = task.task_name
    state = fit(model, [task], {name: schedule[name]}, {name: local[name]} if name in local else {})
    first = task.get_window(0, num_proc=1)
    case, ends = capture_native(task, first, 0)
    prediction = issue(task, case, ends, model.pins, model.provider, CutoffDecision(**state["tasks"][name]))
    return state, received, prediction.values


@pytest.mark.parametrize(("task_local", "cap"), [(False, 5), (False, 18), (True, 22)])
def test_hidden_context_cannot_change_fit_or_qualification(tmp_path, task_local, cap):
    low = native_task(tmp_path, "receiving", hidden=0.0, cap=cap, minimum=2 if cap == 5 else 3)
    high = native_task(tmp_path, "receiving", hidden=60.0, cap=cap, minimum=2 if cap == 5 else 3)
    sibling = native_task(tmp_path, "sibling", shift=-150) if task_local else None
    for a, b in zip(
        low.get_window(0, num_proc=1).get_input_data(), high.get_window(0, num_proc=1).get_input_data(), strict=True
    ):
        assert a.data.table.equals(b.data.table)
    original, first_calls, first_prediction = fitted(low, sibling)
    changed, second_calls, second_prediction = fitted(high, sibling)
    for key in ("tasks", "local_gate", "local_held", "local_used", "counts"):
        assert original[key] == changed[key]
    assert len(first_calls) == len(second_calls) > 8
    assert all(
        a.equals(b)
        for first, second in zip(first_calls, second_calls, strict=True)
        for a, b in zip(first, second, strict=True)
    )
    np.testing.assert_array_equal(first_prediction, second_prediction)
    if task_local:
        assert original["local_used"][low.task_name]
        assert len(original["local_held"][low.task_name]) == 4
    # Sufficient visible history must still support useful calibration.
    if cap >= 18:
        signal = native_task(tmp_path, "receiving", hidden=0.0, visible=60.0, cap=cap)
        useful, _, _ = fitted(signal, sibling)
        original_pool = original["tasks"][low.task_name]["quantile"]["fallback"]
        assert useful["tasks"][low.task_name]["quantile"]["fallback"] != original_pool


def test_staggered_multitarget_eligibility_uses_only_each_visible_prefix(tmp_path):
    tasks = [
        native_task(tmp_path, "staggered", hidden=level, extra_prefix=extra, staggered=True, multitarget=True)
        for level, extra in ((0.0, 0), (70.0, 23))
    ]
    plans, snapshots = [], []
    for task in tasks:
        case, ends = capture_native(task, task.get_window(0, num_proc=1), 0)
        assert len(np.unique(ends)) == 2
        visible = {
            uid: set(np.asarray(times, dtype="datetime64[ns]").astype(np.int64))
            for uid, times in zip(case[3], case[1][task.timestamp_column], strict=True)
        }
        plan = NativeEarlierInputs(task, case, ends)
        plans.append(plan)
        rows = []
        for ordinal in range(len(plan.active_grid)):
            earlier, _, _ = plan.case(ordinal)
            for uid, past, future in zip(
                earlier[3], earlier[1][task.timestamp_column], earlier[2][task.timestamp_column], strict=True
            ):
                used = np.concatenate(
                    (np.asarray(past, dtype="datetime64[ns]"), np.asarray(future, dtype="datetime64[ns]"))
                ).astype(np.int64)
                assert set(used) <= visible[uid]
            rows.append((earlier[3], earlier[4], earlier[1].data.table, earlier[2].data.table))
        snapshots.append(rows)
    np.testing.assert_array_equal(plans[0].grid, plans[1].grid)
    np.testing.assert_array_equal(plans[0].counts, plans[1].counts)
    assert len(snapshots[0]) == len(snapshots[1]) > 0
    for first, second in zip(*snapshots, strict=True):
        assert first[0] == second[0]
        np.testing.assert_array_equal(first[1], second[1])
        assert first[2].equals(second[2]) and first[3].equals(second[3])


def test_cap_below_minimum_keeps_official_inputs_but_has_no_calibration(tmp_path):
    task = native_task(tmp_path, "short_cap", cap=2, minimum=4, staggered=True, multitarget=True)
    past, _ = task.get_window(0, num_proc=1).get_input_data()
    assert len(past) == 2 and all(len(x) == 2 for x in past[task.timestamp_column])
    schedule, local = build_calendars([task], {task.task_name: "source"})
    entry = schedule[task.task_name]
    assert entry["earlier"] == [] and local == {}
    assert entry["shared_disjoint"] == entry["shared_spaced"] == 0
    assert list(earlier_episodes(task, entry["earlier"])) == []
    state, _, values = fitted(task)
    assert state["counts"]["shared"] == state["counts"]["local"] == 0
    assert state["tasks"][task.task_name]["quantile"]["fallback"] == tuple(state["members"])
    assert values.shape == (2, 2, 2, 10) and np.isfinite(values).all()


def test_uncapped_plan_is_identical_to_original_native_input_plan(tmp_path):
    task = native_task(tmp_path, "uncapped", cap=None, staggered=True, multitarget=True)
    first, ends = capture_native(task, task.get_window(0, num_proc=1), 0)
    original, current = EarlierInputs(task, first, ends), NativeEarlierInputs(task, first, ends)
    for field in ("grid", "active_grid", "counts", "first", "last", "official_ends"):
        np.testing.assert_array_equal(getattr(original, field), getattr(current, field))
    for index in (0, len(original.active_grid) // 2, len(original.active_grid) - 1):
        old, old_ends, old_rows = original.case(index)
        new, new_ends, new_rows = current.case(index)
        assert old[3] == new[3] and old[5] == new[5]
        assert old[1].data.table.equals(new[1].data.table)
        assert old[2].data.table.equals(new[2].data.table)
        np.testing.assert_array_equal(old[4], new[4])
        np.testing.assert_array_equal(old_ends, new_ends)
        np.testing.assert_array_equal(old_rows, new_rows)

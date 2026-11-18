"""Generate causal comparison episodes from native FEV input clocks."""

from collections import defaultdict
from copy import copy
from dataclasses import dataclass, replace
from hashlib import sha256
import json

import numpy as np
import pandas as pd
import pyarrow.compute as pc

from .cases import EarlierInputs


MIN_EVIDENCE = 3
HELD = 4


@dataclass(frozen=True)
class NativeEpisode:
    task: object
    row: dict
    case: tuple
    ends: np.ndarray


def capture_native(task, window, origin):
    """Read FEV's typed inputs on its own cutoff and record their original axes."""
    past, future = window.get_input_data()
    shared = {task.id_column, task.timestamp_column, *task.known_dynamic_columns, *task.static_columns}
    if (set(future.column_names) != shared
            or set(past.column_names) != shared | set(task.target_columns) | set(task.past_dynamic_columns)):
        raise ValueError("The provided datasets disagree with the declared FEV roles.")
    ids = list(past[task.id_column])
    if ids != list(future[task.id_column]) or len(ids) != len(set(ids)):
        raise ValueError("Native recipients must have the same unique IDs in both role views.")
    history = past.data.table[task.timestamp_column]
    ends = np.asarray([row.values[-1].as_py() for row in history], dtype="datetime64[ns]")
    delivery = pc.list_flatten(future.data.table[task.timestamp_column]).to_numpy(zero_copy_only=False)
    dates = delivery.astype("datetime64[ns]").reshape(len(ids), task.horizon)
    if not len(ids) or np.isnat(dates).any() or np.isnat(ends).any() or (dates[:, 0] <= ends).any():
        raise ValueError("Each forecast target must follow its own recipient's issue cutoff.")
    meta = {"origin":origin, "series":len(ids), "horizon":task.horizon, "target_columns":task.target_columns,
            "cutoff_spec":window.cutoff, "membership_sha256":sha256(json.dumps(ids,ensure_ascii=False).encode()).hexdigest(),
            "past_end_min_max":[np.datetime_as_string(x,unit="s") for x in (ends.min(),ends.max())],
            "issue_at_ns":int(ends.astype(np.int64).min()),"available_at_ns":int(dates.astype(np.int64).max())}
    return (window,past,future,ids,dates,meta),ends


def provider_frequency(task, past, future):
    """Use native FEV frequency, or infer an unambiguous cadence for a provider."""
    try:
        frequency = task.freq
    except ValueError:
        frequency = None
    if frequency is not None:
        return frequency
    history = past[task.timestamp_column]
    delivery = future[task.timestamp_column]
    observed = set()
    for a,b in zip(history,delivery,strict=True):
        clock = pd.DatetimeIndex(np.concatenate((np.asarray(a)[-8:], np.asarray(b)[:8])))
        inferred = pd.infer_freq(clock) if len(clock) >= 3 else None
        if inferred is None:
            return None
        observed.add(inferred)
    return next(iter(observed)) if len(observed) == 1 else None


class NativeEarlierInputs(EarlierInputs):
    """Enumerate calibration only inside the native first-window history."""

    def __init__(self, task, first_case, ends):
        self.native_max_context = first_case[0].max_context_length
        super().__init__(task,first_case,ends)

    def case(self, index):
        case, ends, positions = super().case(index)
        if self.native_max_context is None:
            return case, ends, positions
        kept = replace(case[0],max_context_length=self.native_max_context)
        current, actual = capture_native(self.task,kept,case[5]["origin"])
        if current[5] != case[5] or not np.array_equal(actual,ends):
            raise ValueError("Applying FEV's context cap changed the earlier forecast target axes.")
        return current,actual,positions


def _earlier_plan(task, first_case, ends):
    window = first_case[0]
    # FEV filters recipients by minimum length before truncating their histories.
    # A valid official recipient can therefore have too little visible history
    # to furnish any earlier calibration episode.
    if window.max_context_length is not None and window.max_context_length < window.min_context_length:
        return None
    return NativeEarlierInputs(task,first_case,ends)


def earlier_episodes(task, rows):
    """Resolve a generated earlier population from the native first receiving window."""
    if not rows:
        return
    first, ends = capture_native(task,task.get_window(0,num_proc=1),0)
    plan = _earlier_plan(task,first,ends)
    if plan is None:
        raise ValueError("The visible first-window history cannot support earlier episodes.")
    for row in rows:
        case, actual, _ = plan.case(row["ordinal"])
        dates = case[4].astype(np.int64)
        if (any(row.get(k) != v for k,v in case[5].items())
                or row["first_target_at_ns"] != dates.min()
                or row["available_at_ns"] != dates.max()):
            raise ValueError("An earlier episode differs from its input-only registration.")
        yield NativeEpisode(task,row,case,actual)


def official_episode(task, row):
    case, ends = capture_native(task,task.get_window(row["origin"],num_proc=1),row["origin"])
    if any(row.get(k) != v for k,v in case[5].items()):
        raise ValueError("A native official evaluation window differs from its input registration.")
    return NativeEpisode(task,row,case,ends)


def _merge(spans):
    merged = []
    for first,last in sorted(spans):
        if first > last:
            raise ValueError("A protected interval ends before its delivery begins.")
        if merged and first <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1],last)
        else:
            merged.append([first,last])
    return merged


def _overlap(first, last, spans):
    if not spans:
        return np.zeros(len(first),dtype=bool)
    ends = np.asarray(spans,dtype=np.int64)
    i = np.searchsorted(ends[:,1],first,side="left")
    return (i < len(ends)) & (ends[np.minimum(i,len(ends)-1),0] <= last)


def _target_bounds(plan):
    diff = np.zeros(len(plan.grid)+1,dtype=np.int64)
    regular = plan.contiguous & (plan.last >= plan.first)
    np.add.at(diff,plan.first[regular],1)
    np.add.at(diff,plan.last[regular]+1,-1)
    live = np.flatnonzero(diff.cumsum()[:-1])
    first = np.full(len(plan.grid),np.iinfo(np.int64).max)
    last = np.full(len(plan.grid),np.iinfo(np.int64).min)
    first[live] = plan.grid[live+1]
    last[live] = plan.grid[live+plan.horizon]
    for index in plan.ragged:
        dates = plan.clocks[plan.offsets[index]:plan.offsets[index+1]]
        offsets = np.arange(plan.minimum-1,len(dates)-plan.horizon-1)
        rows = np.searchsorted(plan.grid,dates[offsets])
        np.minimum.at(first,rows,dates[offsets+1])
        np.maximum.at(last,rows,dates[offsets+plan.horizon])
    a,b = first[plan.active_grid],last[plan.active_grid]
    if (a <= plan.grid[plan.active_grid]).any() or (b < a).any():
        raise ValueError("An eligible earlier issue must have a strictly later target envelope.")
    return a,b


@dataclass(frozen=True)
class Calendar:
    issued: np.ndarray
    first: np.ndarray
    last: np.ndarray
    spaced: np.ndarray

    def disjoint(self, indices):
        indices = np.asarray(indices,dtype=np.int64)
        ordered = np.lexsort((self.issued[indices],self.last[indices]))
        chosen, latest = [], np.iinfo(np.int64).min
        for i in indices[ordered]:
            if self.first[i] > latest:
                chosen.append(int(i))
                latest = self.last[i]
        return np.asarray(chosen,dtype=np.int64)

    def mature(self, candidates, issue):
        candidates = candidates[(self.last[candidates] < issue) & (self.issued[candidates] < issue)]
        return self.disjoint(candidates)[-MIN_EVIDENCE:].tolist()


def _cadence(plan):
    if plan is None:
        empty = np.empty(0,dtype=np.int64)
        return Calendar(empty,empty,empty,empty)
    first,last = _target_bounds(plan)
    clock = plan.grid[plan.active_grid]
    raw = Calendar(clock,first,last,np.empty(0,dtype=np.int64))
    offsets = (plan.active_grid - plan.active_grid[:1]) % plan.horizon
    phase = np.flatnonzero(np.isin(offsets,(0,plan.horizon//2)))
    return replace(raw,spaced=np.union1d(phase,raw.disjoint(np.arange(len(clock),dtype=np.int64))))


def _short(calendar, boundary):
    safe = calendar.spaced[calendar.last[calendar.spaced] < boundary]
    skeleton = calendar.disjoint(safe)
    holds = {int(i):calendar.mature(safe,calendar.issued[i]) for i in skeleton[-HELD:]}
    return int(len(safe)),int(len(skeleton)),holds


def _local(calendar, shared, own, protected, used):
    safe = (calendar.last < own) & ~_overlap(calendar.first,calendar.last,protected)
    fresh = safe & (calendar.last >= shared) & ~_overlap(calendar.first,calendar.last,used)
    admitted = calendar.spaced[safe[calendar.spaced]]
    checked = calendar.spaced[fresh[calendar.spaced]]
    common = calendar.spaced[calendar.last[calendar.spaced] < shared]
    result = {}
    for i in calendar.disjoint(checked)[-HELD:]:
        donors = calendar.mature(admitted,calendar.issued[i])
        older = calendar.mature(common,calendar.issued[i])
        if len(donors) == MIN_EVIDENCE and set(donors) != set(older):
            result[int(i)] = donors
    return int(len(admitted)),int(len(calendar.disjoint(admitted))),result


def _indices(holds):
    return sorted(set(holds) | {donor for donors in holds.values() for donor in donors})


def _rows(plan, calendar, holds, *, namespace, family, shared, own):
    rows = []
    for i in _indices(holds):
        case,ends,_ = plan.case(i)
        meta = case[5]
        dates = case[4].astype(np.int64)
        boundary = shared if namespace == "shared" else own
        stamp = int(ends.astype(np.int64).min())
        if (meta["issue_at_ns"] != calendar.issued[i] or not np.all(ends == ends[0])
                or (dates.min(),dates.max()) != (calendar.first[i],calendar.last[i])
                or not stamp < dates.min() <= dates.max() < boundary):
            raise ValueError("A generated fitting episode changed its native geometry or chronology.")
        record = meta | {"ordinal":i,"first_target_at_ns":int(dates.min()),
                         "role":"validation" if i in holds else "warmup","namespace":namespace,
                         "family":family,"source_family_boundary_ns":shared,"own_boundary_ns":own}
        if i in holds:
            key = "short_warmup_ordinals" if namespace == "shared" else "validation_donor_ordinals"
            record[key] = holds[i]
        elif namespace == "local":
            record["validation_donor_ordinals"] = []
        rows.append(record)
    return rows


def _roles(task):
    return {"known":list(task.known_dynamic_columns),"past_only":list(task.past_dynamic_columns),
            "static":list(task.static_columns)}


def build_calendars(tasks, families, *, progress=None):
    """Return a native shared schedule and thin-support local specs without target outcomes."""
    tasks = tuple(tasks)
    names = {task.task_name for task in tasks}
    if len(names) != len(tasks) or names != set(families):
        raise ValueError("Each distinct benchmark task needs a caller-provided physical lineage.")
    schedule, official_spans, geometry = {}, {}, {}
    for source in tasks:
        task = copy(source)
        name = task.task_name
        official, horizons = [], []
        for idx,window in enumerate(task.iter_windows(num_proc=1)):
            case,ends = capture_native(task,window,idx)
            if idx == 0:
                geometry[name] = _cadence(_earlier_plan(task,case,ends))
            dates = case[4].astype(np.int64)
            native = [(int(lo),int(hi)) for lo,hi in zip(dates.min(axis=1),dates.max(axis=1),strict=True)]
            horizons.append(_merge(native))
            official.append(case[5] | {"first_target_at_ns":int(dates.min()),"namespace":"official"})
        if len(official) != task.num_windows or not official:
            raise ValueError("The native forecast iterator must expose every task window.")
        own = min(row["issue_at_ns"] for row in official)
        schedule[name] = {"official":official,"frequency":task.freq,"roles":_roles(task),"family":families[name],
                          "own_boundary_ns":own,"min_context_length":task.min_context_length,
                          "max_context_length":task.max_context_length}
        official_spans[name] = horizons
        if progress is not None:
            progress("native",name,len(official))

    members = defaultdict(list)
    for name in schedule:
        members[families[name]].append(name)
    used = defaultdict(list)
    shared_plans = {}
    for name,entry in schedule.items():
        siblings = members[families[name]]
        boundary = min(schedule[n]["own_boundary_ns"] for n in siblings)
        entry["source_family_boundary_ns"] = boundary
        calendar = geometry[name]
        safe, count, holds = _short(calendar,boundary)
        shared_plans[name] = holds
        entry["shared_disjoint"] = count
        entry["shared_spaced"] = safe
        used[families[name]].extend((int(calendar.first[i]),int(calendar.last[i])) for i in _indices(holds))

    local = {}
    for source in tasks:
        task = copy(source)
        name = task.task_name
        entry = schedule[name]
        own,shared,family = entry["own_boundary_ns"],entry["source_family_boundary_ns"],families[name]
        siblings = members[family]
        # Only a sibling's targets issued earlier than this task can intersect its fitting horizon.
        protected = _merge(tuple(x) for n in siblings if n != name
                           for w,spans in zip(schedule[n]["official"],official_spans[n],strict=True)
                           if w["issue_at_ns"] < max(schedule[j]["own_boundary_ns"] for j in siblings if j != n)
                           for x in spans)
        first,ends = capture_native(task,task.get_window(0,num_proc=1),0)
        plan = _earlier_plan(task,first,ends)
        calendar = geometry[name]
        holds = shared_plans[name]
        entry["earlier"] = _rows(plan,calendar,holds,namespace="shared",family=family,shared=shared,own=own)
        clean,cards,local_holds = _local(calendar,shared,own,protected,_merge(used[family]))
        entry["local_candidate"] = {"clean_spaced":clean,"clean_disjoint":cards,
                                    "held":local_holds,"required_ordinals":_indices(local_holds)}
        if entry["shared_disjoint"] < MIN_EVIDENCE and local_holds:
            rows = _rows(plan,calendar,local_holds,namespace="local",family=family,shared=shared,own=own)
            local[name] = {"family":family,"shared_boundary":shared,"own_boundary":own,
                           "protected_intervals":protected,"cases":rows,"held":local_holds}
        if progress is not None:
            progress("earlier",name,len(entry["earlier"])+len(local.get(name,{}).get("cases",())))
    return schedule,local

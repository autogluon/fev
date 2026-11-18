"""Bind D-wabi's cutoff policy to issued FEV cases and native earlier losses."""

import json
import warnings
from copy import copy
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
from dwabi import CutoffDecision, CutoffEnsemble, ForecastBudget, ForecastSeries

from .cases import issued_view, typed_panel
from .guarded import ROOT, epsilon


METHOD = "native_cutoff"
CALIBRATION_HISTORY = "native_first_window"
SUPPLEMENTAL = "timeh2o"
BUDGET = ForecastBudget(16_384, 2_000_000)
VERSION = "0.1.1.post1"
LEVELS = tuple(i / 10. for i in range(1, 10))


def input_bindings():
    return {name: sha256(json.dumps(json.loads((ROOT / name).read_bytes()), sort_keys=True).encode()).hexdigest()
            for name in ("provider_pins.json", "dataset_files.json")}


def series_inputs(task, case, ends):
    if tuple(task.quantile_levels) != LEVELS:
        raise ValueError("D-wabi requires the nine native FEV benchmark deciles.")
    return tuple(
        ForecastSeries(int(end.astype(np.int64)), row.history_dates.astype(np.int64), row.history.T,
                       row.future_dates.astype(np.int64), known=row.known, past=row.past, static=row.static)
        for row, end in zip(typed_panel(task, case[1], case[2]), ends, strict=True)
    )


def earlier_cells(task, case, ends):
    if not (ends == ends[0]).all():
        raise ValueError("A registered earlier window must have one common cutoff.")
    timestamp = case[1].data.table.column(task.timestamp_column)
    lengths = pc.list_value_length(timestamp).to_numpy(zero_copy_only=False)
    slots = np.minimum(lengths, BUDGET.context_limit).sum(dtype=np.int64) + len(ends) * task.horizon
    return int(slots) * len(task.target_columns) * (1 + len(task.known_dynamic_columns))


def scoped_case(case, cohort):
    window, past, future, ids, dates, meta = case
    if len(cohort.rows) == len(ids):
        return issued_view(window, past, future), past, future, ids, dates, meta
    rows = list(cohort.rows)
    previous = past.select(rows).flatten_indices()
    ahead = future.select(rows).flatten_indices()
    keys = [ids[i] for i in rows]
    clock = np.datetime_as_string(np.datetime64(cohort.issue_at, "ns"), unit="s")
    scoped = meta | {"issue_at_ns": cohort.issue_at, "series": len(rows), "past_end_min_max": [clock, clock]}
    return issued_view(window, previous, ahead), previous, ahead, keys, dates[rows], scoped


def issue(task, case, ends, members, provider, decision):
    """Issue the frozen wheel policy, loading providers only when the wheel requests them."""
    model_task = copy(task)
    model_task._freq = task.freq
    model_task._full_dataset = None
    cache = {}

    def bind(name):
        def forecast(cohort):
            if cohort.issue_at not in cache:
                # Calls for a cutoff are consecutive. Keep only the currently issued Arrow view.
                cache.clear()
                cache[cohort.issue_at] = scoped_case(case, cohort)
            return provider(name)(model_task, cache[cohort.issue_at])
        return forecast

    ensemble = CutoffEnsemble({name: bind(name) for name in (*members, SUPPLEMENTAL)},
                              epsilon=epsilon(task), supplemental=SUPPLEMENTAL, budget=BUDGET)
    return ensemble.predict(series_inputs(task, case, ends), decision=decision)


def native_sql(task, common, laws):
    metric = next(m for m in task.metrics if m.name == "SQL")
    losses = {}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        for name, law in laws.items():
            inputs = dict(common, y_pred=law[..., 0], q_pred=law[..., 1:])
            score = float(np.mean(metric.compute_scores(
                **inputs, seasonality=task.seasonality, quantile_levels=task.quantile_levels, reduce_axes=(0, 1),
            )["SQL"]))
            losses[name] = score if np.isfinite(score) else None
    if len({score is None for score in losses.values()}) != 1:
        raise ValueError("Earlier providers disagree on native SQL support.")
    return losses


def read_policy(path, members):
    members = tuple(members)
    state = path if isinstance(path, dict) else json.loads(Path(path).read_bytes())
    schedule = state["schedule"]
    if (version("dwabi") != VERSION or state["dwabi_version"] != VERSION or state["method"] != METHOD
            or state["inputs"] != input_bindings() or tuple(state["members"]) != members
            or state["supplemental"] != SUPPLEMENTAL
            or state["budget"] != {"context_limit": BUDGET.context_limit, "max_cells": BUDGET.max_cells}
            or state["tasks"].keys() != schedule.keys() or state.get("task_definitions", {}).keys() != schedule.keys()):
        raise ValueError("Prepare the policy with the same package, checkpoints and FEV inputs.")
    if state.get("calibration_history") != CALIBRATION_HISTORY and any(
        definition.get("max_context_length") is not None for definition in state["task_definitions"].values()
    ):
        raise ValueError("Rebuild the policy using the native context cap for calibration.")
    decisions = {name: CutoffDecision(**rule) for name, rule in state["tasks"].items()}
    for name, decision in decisions.items():
        p, q = decision.point, decision.quantile
        entry = schedule[name]
        gate = state["local_gate"][name]
        if p.accept or p.peer_count or p.comparisons or set(gate["tasks"]) - {name}:
            raise ValueError("Each task must learn its policy from its own training observations.")
        shared = {row["origin"] for row in entry["earlier"] if row["available_at_ns"] < p.issue_at}
        used_local = state["local_used"][name]
        if used_local:
            local = state["local"][name]
            possible = {row["origin"] for row in local["cases"] if row["available_at_ns"] < q.issue_at}
            if (q.issue_at != entry["own_boundary_ns"] or len(q.fallback_origins) != 3 or q.admitted
                    or gate["tasks"] != [name] or not gate["qualified"]):
                raise ValueError("Task-local selection has insufficient registered issue-time evidence.")
        else:
            possible = shared
            if q.issue_at != p.issue_at:
                raise ValueError("The shared quantile and point decisions have different cutoffs.")
        if (p.models != members or q.models != members or q.supplemental != SUPPLEMENTAL
                or p.issue_at != entry["source_family_boundary_ns"]
                or not (set(p.ordinal_origins) | set(q.admitted_origins)) <= shared
                or not set(q.fallback_origins) <= possible):
            raise ValueError("A frozen decision differs from its registered roster or chronology.")
    return schedule, decisions

"""Fit D-wabi from earlier FEV forecasts using each task's declared input contract."""

import json
import math
from argparse import ArgumentParser
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from statistics import fmean
from tempfile import NamedTemporaryFile
from time import perf_counter

import fev
import numpy as np
from dwabi import CutoffDecision, PointHistory, QuantileHistory

from _dwabi_fev.bank import ForecastBank
from _dwabi_fev.cases import predictions
from _dwabi_fev.cutoff import (
    BUDGET, CALIBRATION_HISTORY, METHOD, SUPPLEMENTAL, VERSION, earlier_cells, input_bindings, native_sql,
)
from _dwabi_fev.guarded import ROOT, record
from dwabi.wape import _ordered_deciles
from _dwabi_fev.native_calendar import HELD, MIN_EVIDENCE, build_calendars, earlier_episodes


TOLERANCE = 1e-10


def comparison_inputs(task, case, law):
    common, ids = case[0]._prepare_arrays(task.clean_and_validate_predictions(predictions(task, law)),
                                          task.quantile_levels)
    if list(ids) != case[3]:
        raise ValueError("Earlier scoring changed the forecast recipient order.")
    return common


def clock(row):
    return {"origin": row["origin"], "issued_at": row["issue_at_ns"],
            "first_target_at": row["first_target_at_ns"], "available_at": row["available_at_ns"]}


def local_snapshot(shared, observations, spec, entry, issue):
    original = shared.freeze(issue_at=min(issue, entry["source_family_boundary_ns"]))
    if spec is None or len(original.fallback_origins) >= MIN_EVIDENCE:
        return original, False
    own = QuantileHistory(shared.models, supplemental=shared.supplemental)
    latest = QuantileHistory(shared.models, supplemental=shared.supplemental)
    values = {}
    limit = min(spec["own_boundary"], issue)
    for row, losses in observations:
        first, last = row["first_target_at_ns"], row["available_at_ns"]
        if last >= limit or any(first <= hi and lo <= last for lo, hi in spec["protected_intervals"]):
            continue
        owned = {model: losses[model] for model in own.models}
        own.observe(owned, **clock(row))
        values[row["origin"]] = owned, row
    recent = own.freeze(issue_at=issue).fallback_origins[-MIN_EVIDENCE:]
    if len(recent) < MIN_EVIDENCE:
        return original, False
    for origin in recent:
        losses, row = values[origin]
        latest.observe(losses, **clock(row))
    return latest.freeze(issue_at=issue), True


def matched_scores(task, common, laws, models, pool):
    q = {m: _ordered_deciles(laws[m][..., 1:]) for m in models}
    def combined(members):
        deciles = np.median(np.stack([q[m] for m in members], axis=-1), axis=-1)
        # The native SQL reads only the quantile columns.
        return np.concatenate((laws[models[0]][..., :1], deciles), axis=-1)
    return native_sql(task, common, {"reference": combined(models), "candidate": combined(pool)})


def qualification_as_of(validation, families, issue, models):
    """Qualify the current task's local shortlist on its four earlier validation forecasts."""
    if len(validation) > 1:
        raise ValueError("Quantile qualification accepts earlier outcomes from one task at a time.")
    blocks = {name: rows for name, rows in validation.items() if len(rows) == HELD and
              all(row["available_at_ns"] < issue for row in rows)}
    result = {"qualified": False, "issue_at_ns": issue, "tasks": list(blocks),
              "held_cases": sum(len(rows) for rows in blocks.values())}
    if not blocks:
        return result
    ratios, wins = {}, {"reference": [], "candidate": []}
    for name, rows in blocks.items():
        if any(row[arm] is None for row in rows for arm in wins):
            return result
        scores = {arm: fmean(row[arm] for row in rows) for arm in wins}
        if min(scores.values()) <= 0:
            return result
        ratios[name] = math.log(scores["candidate"] / scores["reference"])
        standalone = [fmean(row["standalone"][m] for row in rows) for m in models]
        for arm, score in scores.items():
            wins[arm].extend(.5 if math.isclose(score, other, rel_tol=TOLERANCE, abs_tol=TOLERANCE)
                             else float(score < other) for other in standalone)
    grouped = {f: math.exp(fmean(v for name, v in ratios.items() if families[name] == f))
               for f in sorted({families[name] for name in ratios})}
    ratio = math.exp(fmean(ratios.values()))
    rates = {arm: fmean(values) for arm, values in wins.items()}
    return result | {"SQL_ratio": ratio, "family_SQL_ratios": grouped, "standalone_win_rates": rates,
                     "qualified": ratio < 1 - TOLERANCE and all(v < 1 - TOLERANCE for v in grouped.values())
                     and rates["candidate"] + TOLERANCE >= rates["reference"]}


def registered_episodes(task, entry, local, models):
    members = tuple(models)
    for episode in earlier_episodes(task, entry["earlier"]):
        active = (SUPPLEMENTAL,) if earlier_cells(task, episode.case, episode.ends) <= BUDGET.max_cells else ()
        yield episode, members + active
    if local:
        for episode in earlier_episodes(task, local["cases"]):
            yield episode, members


def fit(model, tasks, schedule, local, bank=None):
    """Fit on issued shared and local windows. Official outcomes never enter this function."""
    if version("dwabi") != VERSION:
        raise ValueError(f"This study uses dwabi=={VERSION}.")
    started = perf_counter()
    models = tuple(model.pins)
    families = {name: entry["family"] for name, entry in schedule.items()}
    validation, task_definitions, decisions, local_used, gates = {}, {}, {}, {}, {}
    counts = {k: 0 for k in ("shared", "local", "defined_point", "supplemental")}
    for task in tasks:
        name = task.task_name
        point = PointHistory(models)
        task_definitions[name] = task.to_dict()
        model.check_data(task)
        entry = schedule[name]
        shared = QuantileHistory(point.models, supplemental=SUPPLEMENTAL)
        evidence, held = [], []
        for episode, members in registered_episodes(task, entry, local.get(name), point.models):
            row, case = episode.row, episode.case
            laws = (model._forecast(task, case, episode.ends, members) if bank is None else
                    bank.forecast(model, task, row, case, episode.ends, members))
            if tuple(laws) != members:
                raise ValueError("Earlier forecasts have an incomplete or reordered provider roster.")
            common = comparison_inputs(task, case, laws[members[0]])
            losses = native_sql(task, common, laws)
            counts[row["namespace"]] += 1
            if row["namespace"] == "shared":
                counts["defined_point"] += record(point, task, families[name], row, common["y_true"], laws)
                counts["supplemental"] += SUPPLEMENTAL in members
                shared.observe(losses, **clock(row))
            else:
                if row["role"] == "validation":
                    before, _ = local_snapshot(shared, evidence, local[name], entry, row["issue_at_ns"])
                    held.append({**matched_scores(task, common, laws, point.models, before.fallback),
                                 "origin": row["origin"], "available_at_ns": row["available_at_ns"],
                                 "standalone": losses, "donors": list(before.fallback_origins)})
                evidence.append((row, losses))
        if held:
            validation[name] = held
        base = shared.freeze(issue_at=entry["source_family_boundary_ns"])
        candidate, eligible = local_snapshot(shared, evidence, local.get(name), entry, entry["own_boundary_ns"])
        gate = qualification_as_of({name: held}, {name: families[name]}, entry["own_boundary_ns"], models)
        local_used[name] = eligible and gate["qualified"]
        gates[name] = gate
        point_action = point.freeze(name, families[name], issue_at=entry["source_family_boundary_ns"])
        decisions[name] = asdict(CutoffDecision(point_action, candidate if local_used[name] else base))
        print(f"{name}: earlier comparisons fitted", flush=True)
    if decisions.keys() != schedule.keys():
        raise ValueError("Every native task needs one fitted decision.")
    return {"method": METHOD, "dwabi_version": VERSION, "inputs": input_bindings(), "members": list(models),
            "supplemental": SUPPLEMENTAL, "budget": asdict(BUDGET), "tasks": decisions, "schedule": schedule,
            "task_definitions": task_definitions, "calibration_history": CALIBRATION_HISTORY,
            "local": local, "local_used": local_used, "local_gate": gates, "local_held": validation,
            "counts": counts, "seconds": perf_counter() - started}


def main():
    from model import DWabiFEVModel

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("--families", type=Path, default=ROOT / "physical_families.json",
                        help="Source lineage used to exclude overlapping forecast intervals. Keys identify tasks.")
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--forecast-bank", type=Path)
    parser.add_argument("--issue-member", help="Generate one provider's earlier forecasts into the bank.")
    args = parser.parse_args()
    if args.issue_member and (args.output is not None or args.forecast_bank is None):
        parser.error("For --issue-member, specify --forecast-bank and omit the policy output.")
    if not args.issue_member and (args.output is None or args.output.exists()):
        parser.error("Specify a fresh policy output path.")
    tasks = fev.Benchmark.from_yaml(str(args.benchmark)).tasks
    families = json.loads(args.families.read_bytes())
    model = DWabiFEVModel(checkpoint_root=args.checkpoint_root, device=args.device)
    for task in tasks:
        model.check_data(task)
    schedule, local = build_calendars(tasks, {t.task_name: families[t.task_name] for t in tasks})
    bank = ForecastBank(args.forecast_bank, input_bindings()) if args.forecast_bank else None
    if args.issue_member:
        if args.issue_member not in model.provider_pins:
            parser.error("Choose a provider in provider_pins.json.")
        for task in tasks:
            for episode, members in registered_episodes(task, schedule[task.task_name], local.get(task.task_name), model.pins):
                if args.issue_member in members:
                    bank.forecast(model, task, episode.row, episode.case, episode.ends, (args.issue_member,))
        return
    result = fit(model, tasks, schedule, local, bank)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with NamedTemporaryFile(dir=args.output.parent, mode="w", encoding="utf-8", suffix=".json", delete=False) as stream:
            temp = Path(stream.name)
            json.dump(result, stream, allow_nan=False)
        args.output.hardlink_to(temp)
    finally:
        if temp is not None:
            temp.unlink()


if __name__ == "__main__":
    main()

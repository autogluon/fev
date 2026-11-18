"""Regression checks for issue-time fields, native windows and prior-only selection."""

import json
from datetime import datetime, timedelta
from types import SimpleNamespace

import datasets
import numpy as np
import pytest
from fev import Task
from fev.task import EvaluationWindow

from model import DWabiFEVModel
from prepare_cutoff import fit, qualification_as_of
from _dwabi_fev.cases import typed_panel
from _dwabi_fev.native_calendar import capture_native


def example(known_name="driver", past_name="sensor", task_name="example", delta_target=0, delta_known=0):
    start = datetime(2020, 1, 1)
    clock = [[start + timedelta(days=i + offset) for i in range(9)] for offset in (0, 3)]
    y = [0, 2, 1, 5, 0, None, 4, 7 + delta_target, 18 + delta_target]
    known = [None, 2, 2, 2, 2, 2, 2, 4 + delta_known, 6 + delta_known]
    past = [None, 3, 3, 3, 3, 3, 3, 99 + delta_target, 100 + delta_target]
    data = datasets.Dataset.from_dict({"id": ["a", "b"], "timestamp": clock, "y": [y, y],
                                        known_name: [known, known], past_name: [past, past], "constant": [3, 4]})
    schema = dict(id_column="id", timestamp_column="timestamp", target_columns=["y"],
                  known_dynamic_columns=[known_name], past_dynamic_columns=[past_name], static_columns=["constant"])
    t = SimpleNamespace(task_name=task_name, freq="D", horizon=2, quantile_levels=list(np.arange(1, 10) / 10), **schema)
    w = EvaluationWindow(full_dataset=data, cutoff=-2, horizon=2, min_context_length=2, max_context_length=5, **schema)
    return t, capture_native(t, w, 0)


def issued_values(**kwargs):
    task, (case, ends) = example(**kwargs)
    runner = DWabiFEVModel(device="cpu")
    def provider(task, issued):
        window, past, future, ids, dates, _ = issued
        assert task._full_dataset is None and window.full_dataset is None
        assert len(ids) == 1  # The native recipients have different actual issue timestamps.
        assert set(future.column_names) == {"id", "timestamp", "constant", *task.known_dynamic_columns}
        item, = typed_panel(task, past, future)
        assert len(item.history_dates) == 5
        assert len(next(iter(item.past.values()))) == 5
        assert np.isnan(item.history[0, -2])
        future_driver = next(iter(item.known.values()))[-task.horizon:]
        return np.broadcast_to(future_driver[None, :, None, None], (1, dates.shape[1], 1, 10)).copy()
    runner.providers["probe"] = provider
    return runner._forecast(task, case, ends, ("probe",))["probe"]


def test_native_future_roles_and_target_visibility():
    original = issued_values()
    np.testing.assert_array_equal(original[:, :, 0, 0], [[4, 6], [4, 6]])
    np.testing.assert_array_equal(issued_values(delta_target=1000), original)
    np.testing.assert_array_equal(issued_values(delta_known=100), original + 100)
    np.testing.assert_array_equal(issued_values(known_name="x", past_name="z", task_name="unseen"), original)


def test_changed_native_metric_contract_cannot_reuse_a_fitted_policy():
    task = Task(dataset_path="example", task_name="example", seasonality=7, eval_metric="MASE")
    runner = DWabiFEVModel(device="cpu")
    runner._prepared = True
    runner.decisions[task.task_name] = object()
    runner.schedule[task.task_name] = {}
    runner.task_definitions[task.task_name] = task.to_dict()
    task.seasonality = 1
    with pytest.raises(ValueError, match="task definition differs"):
        runner.fit_predict(task)


def test_stock_fit_predict_and_prepared_policy_issue_the_same_known_driver(tmp_path, monkeypatch):
    monkeypatch.delenv("DWABI_FEV_POLICY", raising=False)
    start = datetime(2020, 1, 1)
    driver = list(map(float, range(25)))
    data = datasets.Dataset.from_dict({"id": ["series"],
        "timestamp": [[start + timedelta(days=i) for i in range(25)]],
        "target": [driver], "driver": [driver]})
    data_file = tmp_path / "panel.parquet"
    data.to_parquet(str(data_file))
    task = Task(dataset_path=str(data_file), task_name="task", horizon=2, num_windows=2,
                seasonality=1, min_context_length=3, known_dynamic_columns=["driver"],
                eval_metric="SQL", extra_metrics=["MASE", {"name": "WAPE", "epsilon": 1},
                {"name": "WQL", "epsilon": 1}], quantile_levels=[i / 10 for i in range(1, 10)])
    suite, families = tmp_path / "suite.yaml", tmp_path / "families.json"
    suite.write_text(json.dumps({"tasks": [task.to_dict()]}))
    families.write_text(json.dumps({"task": "source"}))

    def probe(task, issued):
        forecast = np.asarray(issued[2]["driver"])
        return np.broadcast_to(forecast[:, :, None, None], (*forecast.shape, 1, 10)).copy()

    model = DWabiFEVModel(device="cpu", benchmark=suite, families=families)
    model.providers = dict.fromkeys(model.provider_pins, probe)
    results = model.fit_predict(task)
    schedule, local, _ = model._calendar
    fitted = fit(model, [task], schedule, local)
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(fitted, allow_nan=False))
    prepared = DWabiFEVModel(device="cpu", policy_path=path)
    prepared.providers = model.providers.copy()
    replay = prepared.fit_predict(task)
    for index, (issued, saved) in enumerate(zip(results, replay, strict=True)):
        assert issued["target"].data.table.equals(saved["target"].data.table)
        np.testing.assert_array_equal(issued["target"]["0.5"][0], driver[21 + 2 * index:23 + 2 * index])
    assert task.evaluation_summary(results, model_name="D-wabi")["SQL"] == 0


def test_local_gate_uses_only_complete_blocks_mature_by_the_actual_cutoff():
    models = ("m1", "m2", "m3")
    def earlier(future, scores):
        return [{"available_at_ns": future + i, "reference": 2.0, "candidate": scores,
                 "standalone": dict.fromkeys(models, 1.5)} for i in range(4)]
    first = {"a": earlier(1, 1.0)}
    second = {"b": earlier(100, 5.0)}
    families = {"a": "g1", "b": "g2"}
    assert not qualification_as_of(first, families, 4, models)["qualified"]
    earlier_gate = qualification_as_of(first, families, 103, models)
    assert earlier_gate["tasks"] == ["a"] and earlier_gate["qualified"]
    assert not qualification_as_of(second, families, 104, models)["qualified"]
    with pytest.raises(ValueError, match="one task at a time"):
        qualification_as_of(first | second, families, 104, models)

"""Forecast cache identity follows observable provider inputs, never outcomes."""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, time, timedelta
from importlib.metadata import version
from threading import Barrier

import datasets
import numpy as np
import pyarrow as pa
import pytest
from _dwabi_fev.bank import ForecastBank, array_hash, column_hash
from _dwabi_fev.native_calendar import capture_native

from fev import Task


def example(change=None):
    start = datetime.fromisoformat("2020-01-01")
    data = {
        "id": ["a", "b"],
        "timestamp": [[start + timedelta(days=i) for i in range(9)]] * 2,
        "target": [
            [1.0, 2.0, 3.0, None, 5.0, 6.0, 7.0, 80.0, 90.0],
            [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 180.0, 190.0],
        ],
        "driver": [[1.0, 2.0, 3.0, None, 5.0, 6.0, 7.0, 8.0, 9.0]] * 2,
        "category": [["a", "b", "a", None, "a", "b", "a", "b", "c"]] * 2,
        "sensor": [[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 800.0, 900.0]] * 2,
        "static": ["first", "second"],
    }
    if change == "target":
        data["target"][0][4] = 55.0
    elif change == "mask":
        data["target"][0][3] = np.nan
    elif change == "historic_known":
        data["driver"][0] = [1.0, 2.0, 3.0, None, 55.0, 6.0, 7.0, 8.0, 9.0]
    elif change == "future_known":
        data["driver"][0] = [1.0, 2.0, 3.0, None, 5.0, 6.0, 7.0, 88.0, 9.0]
    elif change == "category":
        data["category"][0] = ["a", "b", "a", None, "a", "b", "a", "new", "c"]
    elif change == "past_only":
        data["sensor"][0] = [10.0, 20.0, 30.0, 40.0, 55.0, 60.0, 70.0, 800.0, 900.0]
    elif change == "static":
        data["static"][0] = "different"
    elif change == "outcomes":
        data["target"][0][7:] = [8000.0, 9000.0]
        data["sensor"][0] = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 8000.0, 9000.0]
    task = Task(
        dataset_path="unused.parquet",
        task_name="example",
        horizon=2,
        min_context_length=2,
        max_context_length=5,
        known_dynamic_columns=["driver", "category"],
        past_dynamic_columns=["sensor"],
        static_columns=["static"],
        seasonality=1,
        quantile_levels=[i / 10 for i in range(1, 10)],
    )
    task._full_dataset = datasets.Dataset.from_dict(data).with_format("numpy")
    task._freq = "D"
    case, ends = capture_native(task, task.get_window(0, num_proc=1), 0)
    return task, case[-1] | {"namespace": "shared"}, case, ends


class Provider:
    device = "cpu"

    def __init__(self, barrier=None, marker=1.0):
        self.calls = 0
        self.barrier, self.marker = barrier, marker
        self.provider_pins = {"member": {"package": "numpy", "version": version("numpy")}}

    def _forecast(self, task, case, ends, members):
        self.calls += 1
        if self.barrier is not None:
            self.barrier.wait(timeout=10)
        shape = (*case[4].shape, len(task.target_columns), len(task.quantile_levels) + 1)
        return {name: np.full(shape, self.marker) for name in members}


def issue(bank, model, inputs):
    return bank.forecast(model, *inputs, ("member",))["member"]


@pytest.mark.parametrize(
    "change", ["target", "mask", "historic_known", "future_known", "category", "past_only", "static"]
)
def test_changed_observable_values_or_masks_cannot_reuse_receipt(tmp_path, change):
    bank, model = ForecastBank(tmp_path, {"source": "fixed"}), Provider()
    issue(bank, model, example())
    original = (tmp_path / "member/example/0.npz").read_bytes()
    with pytest.raises(ValueError):
        issue(bank, model, example(change))
    assert model.calls == 1
    assert (tmp_path / "member/example/0.npz").read_bytes() == original


@pytest.mark.parametrize("setting", ["frequency", "task", "cap", "window_minimum", "provider_pin", "dtype"])
def test_effective_input_and_provider_settings_are_bound(tmp_path, setting):
    bank, model = ForecastBank(tmp_path, {}), Provider()
    inputs = example()
    issue(bank, model, inputs)
    task, row, case, ends = example()
    if setting == "frequency":
        task._freq = "h"
    elif setting == "task":
        task.seasonality = 7
    elif setting in ("cap", "window_minimum"):
        window = replace(case[0], **({"max_context_length": 10} if setting == "cap" else {"min_context_length": 1}))
        case = (window, *case[1:])
    elif setting == "provider_pin":
        model.provider_pins["member"]["revision"] = "another-checkpoint"
    else:
        past = case[1].cast_column("driver", datasets.Sequence(datasets.Value("int64")))
        future = case[2].cast_column("driver", datasets.Sequence(datasets.Value("int64")))
        case = (case[0], past, future, *case[3:])
    with pytest.raises(ValueError):
        issue(bank, model, (task, row, case, ends))
    assert model.calls == 1


def test_cache_does_not_bind_unobserved_target_or_past_covariate_outcomes(tmp_path):
    bank, model = ForecastBank(tmp_path, {}), Provider()
    expected = issue(bank, model, example())
    np.testing.assert_array_equal(issue(bank, model, example("outcomes")), expected)
    assert model.calls == 1


def test_logical_dataset_views_ignore_hidden_rows_and_chunk_boundaries(tmp_path):
    bank, model = ForecastBank(tmp_path, {}), Provider()

    def view(inputs, chunked=False):
        task, row, case, ends = inputs
        previous, future = case[1].select([1]), case[2].select([1])
        if chunked:

            def rebuild(dataset):
                table = dataset.with_format("arrow")[:]
                return datasets.Dataset(pa.Table.from_batches(table.to_batches(max_chunksize=1))).with_format("numpy")

            previous, future = rebuild(previous), rebuild(future)
        case = (case[0], previous, future, ["b"], case[4][1:], case[5])
        return task, row | {"series": 1}, case, ends[1:]

    expected = issue(bank, model, view(example()))
    np.testing.assert_array_equal(issue(bank, model, view(example("target"))), expected)
    np.testing.assert_array_equal(issue(bank, model, view(example(), chunked=True)), expected)
    assert model.calls == 1


def test_chunk_layout_and_hidden_sliced_child_buffers_do_not_change_identity(tmp_path):
    bank, model = ForecastBank(tmp_path, {}), Provider()
    inputs = example()
    expected = issue(bank, model, inputs)
    task, row, case, ends = inputs

    def rebuild(dataset, hidden):
        table = dataset.with_format("arrow")[:]
        if hidden:
            # Each list slice now retains hidden prefix/suffix child buffers.
            table = pa.concat_tables([table, table, table]).combine_chunks().slice(len(table), len(table))
        else:
            table = pa.Table.from_batches(table.to_batches(max_chunksize=1))
        return datasets.Dataset(table).with_format("numpy")

    for hidden in (False, True):
        alternate = (case[0], rebuild(case[1], hidden), rebuild(case[2], hidden), *case[3:])
        np.testing.assert_array_equal(issue(bank, model, (task, row, alternate, ends)), expected)
    assert model.calls == 1


def test_datetime_array_identity_is_stable_across_native_byte_order():
    clock = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]")
    assert array_hash(clock) == array_hash(clock.astype(clock.dtype.newbyteorder(">")))
    assert array_hash(np.array([1, 2], dtype="<i8")) == array_hash(np.array([1, 2], dtype=">i8"))
    assert array_hash(np.array([1, 2], dtype="i8")) != array_hash(np.array([1, 2], dtype="f8"))


def test_arrow_digest_ignores_hidden_null_list_values_and_physical_offsets():
    logical = pa.array([[1.0, None], [], None, [np.nan, 2.0]])
    hidden = pa.ListArray.from_arrays(
        pa.array([0, 2, 2, 4, 6]),
        pa.array([1.0, None, 777.0, 888.0, np.nan, 2.0]),
        mask=pa.array([False, False, True, False]),
    )
    sliced = pa.concat_arrays([logical, logical, logical]).slice(4, 4)
    chunked = pa.chunked_array([logical.slice(0, 1), logical.slice(1)])
    expected = column_hash(pa.chunked_array([logical]))
    assert column_hash(pa.chunked_array([hidden])) == expected
    assert column_hash(pa.chunked_array([sliced])) == expected
    assert column_hash(chunked) == expected


@pytest.mark.parametrize("dtype", [pa.time32("s"), pa.time64("us")])
def test_time_values_use_stable_storage_across_chunks_and_repeated_calls(dtype):
    values = pa.array([time(3, 4), None], type=dtype)
    chunked = pa.chunked_array([values.slice(0, 1), values.slice(1)])
    expected = column_hash(pa.chunked_array([values]))
    assert column_hash(chunked) == expected
    assert column_hash(pa.chunked_array([pa.array([time(3, 4), None], type=dtype)])) == expected
    assert column_hash(pa.chunked_array([pa.array([time(3, 5), None], type=dtype)])) != expected


def test_temporal_intervals_are_rejected_instead_of_hashing_python_objects():
    values = pa.array([(1, 2, 3)], type=pa.month_day_nano_interval())
    with pytest.raises(TypeError):
        column_hash(pa.chunked_array([values]))


def test_pre_versioned_receipt_is_rejected_without_rewriting_it(tmp_path):
    bank, model = ForecastBank(tmp_path, {}), Provider()
    issue(bank, model, example())
    path = tmp_path / "member/example/0.npz"
    with np.load(path, allow_pickle=False) as saved:
        law, metadata = saved["law"], json.loads(str(saved["metadata"].item()))
    legacy_fields = {
        "inputs",
        "member",
        "task",
        "issue",
        "cutoffs_sha256",
        "lead_times_sha256",
        "known",
        "past_only",
        "static",
        "deciles",
        "law_sha256",
        "producer",
    }
    metadata = {name: value for name, value in metadata.items() if name in legacy_fields}
    np.savez_compressed(path, law=law, metadata=json.dumps(metadata))
    original = path.read_bytes()
    with pytest.raises(ValueError, match="(?i)(fresh|version|binding|cache)"):
        issue(bank, model, example())
    assert path.read_bytes() == original and model.calls == 1


@pytest.mark.parametrize("different", [False, True])
def test_atomic_concurrent_publication_verifies_the_winning_receipt(tmp_path, different):
    barrier = Barrier(2)
    models = [Provider(barrier, marker=1.0), Provider(barrier, marker=2.0)]
    inputs = [example(), example("target" if different else None)]

    def run(index):
        try:
            return issue(ForecastBank(tmp_path, {}), models[index], inputs[index])
        except ValueError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(run, range(2)))
    errors = [value for value in results if isinstance(value, ValueError)]
    assert len(errors) == int(different)
    if not different:
        np.testing.assert_array_equal(*results)
    with np.load(tmp_path / "member/example/0.npz", allow_pickle=False) as saved:
        assert np.isfinite(saved["law"]).all()
    assert not list(tmp_path.rglob("*.lock"))

import datasets
import numpy as np
import pandas as pd
import pytest
from datasets import Dataset

import fev
from fev.utils import generate_univariate_targets_from_multivariate, past_future_split


def test_when_dataset_info_is_changed_then_dataset_fingerprint_doesnt_change():
    ds = datasets.load_dataset("autogluon/chronos_datasets", "monash_m1_yearly", split="train")
    old_fingerprint = fev.utils.generate_fingerprint(ds)
    ds._info = datasets.DatasetInfo("New custom description")
    new_fingerprint = fev.utils.generate_fingerprint(ds)
    assert isinstance(old_fingerprint, str)
    assert isinstance(new_fingerprint, str)
    assert old_fingerprint == new_fingerprint


def test_when_dataset_dict_provided_to_generate_fingerprint_then_exception_is_raised():
    ds_dict = datasets.load_dataset("autogluon/chronos_datasets", "monash_m1_yearly")
    with pytest.raises(ValueError, match="datasets.Dataset"):
        fev.utils.generate_fingerprint(ds_dict)


def test_when_sequence_col_entries_have_different_lengths_then_validate_dataset_raises_an_error():
    N = 3
    ds = datasets.Dataset.from_list(
        [
            {"id": "A", "timestamp": pd.date_range("2020", freq="D", periods=N), "target": list(range(N))},
            {"id": "B", "timestamp": pd.date_range("2020", freq="D", periods=N), "target": list(range(N + 1))},
        ]
    )
    with pytest.raises(AssertionError, match="Lengths of entries in"):
        fev.utils.validate_time_series_dataset(ds)


def _make_ts_record(id: str, length: int, start: str = "2020-01-01", freq: str = "D", static_val: int | None = None):
    """Create a single time series record."""
    record = {
        "id": id,
        "timestamp": pd.date_range(start, freq=freq, periods=length),
        "target": list(range(length)),
    }
    if static_val is not None:
        record["static"] = static_val
    return record


# Cutoff and slicing tests


@pytest.mark.parametrize(
    "length, cutoff, horizon, min_ctx, expected_past, expected_future",
    [
        (10, 5, 3, 1, [0, 1, 2, 3, 4], [5, 6, 7]),  # positive cutoff middle
        (10, -3, 2, 1, [0, 1, 2, 3, 4, 5, 6], [7, 8]),  # negative cutoff
        (10, 2, 3, 2, [0, 1], [2, 3, 4]),  # cutoff near start
        (10, 8, 2, 1, [0, 1, 2, 3, 4, 5, 6, 7], [8, 9]),  # cutoff near end
        (10, 5, 5, 1, [0, 1, 2, 3, 4], [5, 6, 7, 8, 9]),  # horizon equals remaining
        (5, 1, 2, 1, [0], [1, 2]),  # single element past
        (5, 0, 3, 0, [], [0, 1, 2]),  # zero cutoff with zero min_context
    ],
)
def test_when_integer_cutoff_provided_then_data_split_correctly(
    length, cutoff, horizon, min_ctx, expected_past, expected_future
):
    ds = Dataset.from_list([_make_ts_record("A", length)])
    past, future = past_future_split(
        ds, timestamp_column="timestamp", cutoff=cutoff, horizon=horizon, min_context_length=min_ctx
    )

    assert len(past) == 1
    assert list(past[0]["target"]) == expected_past
    assert list(future[0]["target"]) == expected_future


@pytest.mark.parametrize(
    "length, cutoff, horizon, expected_past, expected_future",
    [
        (10, "2020-01-05", 3, [0, 1, 2, 3, 4], [5, 6, 7]),  # cutoff on datapoint
        (5, "2020-01-02T12:00:00", 2, [0, 1], [2, 3]),  # cutoff between datapoints
    ],
)
def test_when_timestamp_cutoff_provided_then_data_split_correctly(
    length, cutoff, horizon, expected_past, expected_future
):
    ds = Dataset.from_list([_make_ts_record("A", length, start="2020-01-01")])
    past, future = past_future_split(
        ds, timestamp_column="timestamp", cutoff=cutoff, horizon=horizon, min_context_length=1
    )

    assert len(past) == 1
    assert list(past[0]["target"]) == expected_past
    assert list(future[0]["target"]) == expected_future


# Filtering tests


@pytest.mark.parametrize(
    "lengths, cutoff, horizon, min_ctx, expected_num_items, expected_ids",
    [
        ([10, 10], 5, 3, 3, 2, ["A", "B"]),  # all sufficient
        ([10, 3], 5, 2, 4, 1, ["A"]),  # B insufficient context
        ([10, 6], 5, 3, 1, 1, ["A"]),  # B insufficient future
        ([20, 5, 18, 11], 10, 5, 8, 2, ["A", "C"]),  # mixed filtering
        ([5, 5], 3, 10, 1, 0, []),  # all filtered
        ([10], 5, 2, 6, 0, []),  # min_context exceeds available
    ],
)
def test_when_series_filtered_then_correct_items_remain(
    lengths, cutoff, horizon, min_ctx, expected_num_items, expected_ids
):
    ids = ["A", "B", "C", "D"][: len(lengths)]
    ds = Dataset.from_list([_make_ts_record(id, length) for id, length in zip(ids, lengths)])
    past, future = past_future_split(
        ds, timestamp_column="timestamp", cutoff=cutoff, horizon=horizon, min_context_length=min_ctx
    )

    assert len(past) == expected_num_items
    assert len(future) == expected_num_items
    if expected_num_items > 0:
        remaining_ids = [past[i]["id"] for i in range(len(past))]
        assert set(remaining_ids) == set(expected_ids)


# max_context_length tests


@pytest.mark.parametrize(
    "length, cutoff, horizon, max_ctx, expected_past",
    [
        (20, 15, 3, 5, [10, 11, 12, 13, 14]),  # limited by max_context
        (10, 5, 3, 100, [0, 1, 2, 3, 4]),  # max_context exceeds available
        (20, 15, 3, None, list(range(15))),  # None returns all
    ],
)
def test_when_max_context_length_set_then_past_limited_correctly(length, cutoff, horizon, max_ctx, expected_past):
    ds = Dataset.from_list([_make_ts_record("A", length)])
    past, future = past_future_split(
        ds,
        timestamp_column="timestamp",
        cutoff=cutoff,
        horizon=horizon,
        min_context_length=1,
        max_context_length=max_ctx,
    )

    assert list(past[0]["target"]) == expected_past


# Multiple columns tests


def test_when_multiple_sequence_columns_present_then_all_are_sliced():
    ds = Dataset.from_list(
        [
            {
                "id": "A",
                "timestamp": pd.date_range("2020", freq="D", periods=10),
                "target": list(range(10)),
                "feature1": list(range(100, 110)),
                "feature2": list(range(200, 210)),
            }
        ]
    )
    past, future = past_future_split(ds, timestamp_column="timestamp", cutoff=5, horizon=3, min_context_length=1)

    assert list(past[0]["target"]) == [0, 1, 2, 3, 4]
    assert list(past[0]["feature1"]) == [100, 101, 102, 103, 104]
    assert list(past[0]["feature2"]) == [200, 201, 202, 203, 204]
    assert list(future[0]["target"]) == [5, 6, 7]
    assert list(future[0]["feature1"]) == [105, 106, 107]
    assert list(future[0]["feature2"]) == [205, 206, 207]


def test_when_static_columns_present_then_they_are_preserved():
    ds = Dataset.from_list(
        [
            _make_ts_record("A", 10, static_val=42),
            _make_ts_record("B", 10, static_val=99),
        ]
    )
    past, future = past_future_split(ds, timestamp_column="timestamp", cutoff=5, horizon=3, min_context_length=1)

    assert past[0]["static"] == 42
    assert past[1]["static"] == 99
    assert future[0]["static"] == 42
    assert future[1]["static"] == 99


def test_when_string_dynamic_column_present_then_it_is_sliced():
    ds = Dataset.from_list(
        [
            {
                "id": "A",
                "timestamp": pd.date_range("2020", freq="D", periods=6),
                "target": [1, 2, 3, 4, 5, 6],
                "category": ["a", "b", "c", "d", "e", "f"],
            }
        ]
    )
    past, future = past_future_split(ds, timestamp_column="timestamp", cutoff=4, horizon=2, min_context_length=1)

    assert list(past[0]["category"]) == ["a", "b", "c", "d"]
    assert list(future[0]["category"]) == ["e", "f"]


def test_when_string_static_column_present_then_it_is_preserved():
    ds = Dataset.from_list(
        [
            {
                "id": "A",
                "timestamp": pd.date_range("2020", freq="D", periods=6),
                "target": [1, 2, 3, 4, 5, 6],
                "region": "north",
            },
            {
                "id": "B",
                "timestamp": pd.date_range("2020", freq="D", periods=6),
                "target": [1, 2, 3, 4, 5, 6],
                "region": "south",
            },
        ]
    )
    past, future = past_future_split(ds, timestamp_column="timestamp", cutoff=4, horizon=2, min_context_length=1)

    assert past[0]["region"] == "north"
    assert past[1]["region"] == "south"
    assert future[0]["region"] == "north"
    assert future[1]["region"] == "south"


# Variable length tests


@pytest.mark.parametrize(
    "lengths, starts, cutoff, horizon, expected_past_lengths",
    [
        # integer cutoff: all get same cutoff index
        ([15, 20, 10], ["2020-01-01"] * 3, 7, 2, [7, 7, 7]),
        # negative cutoff: past length varies by series length
        ([10, 15, 8], ["2020-01-01"] * 3, -3, 2, [7, 12, 5]),
        # timestamp cutoff with different start dates
        ([10, 15, 10], ["2020-01-01", "2020-01-01", "2020-01-03"], "2020-01-07", 2, [7, 7, 5]),
    ],
)
def test_when_series_have_variable_lengths_then_past_lengths_computed_correctly(
    lengths, starts, cutoff, horizon, expected_past_lengths
):
    ds = Dataset.from_list(
        [_make_ts_record(f"S{i}", length, start) for i, (length, start) in enumerate(zip(lengths, starts))]
    )
    past, future = past_future_split(
        ds, timestamp_column="timestamp", cutoff=cutoff, horizon=horizon, min_context_length=1
    )

    assert len(past) == len(lengths)
    for i, expected_len in enumerate(expected_past_lengths):
        assert len(past[i]["target"]) == expected_len


# Other tests


def test_when_many_series_provided_then_all_processed_correctly():
    records = [_make_ts_record(f"series_{i}", length=50 + i % 10) for i in range(100)]
    ds = Dataset.from_list(records)
    past, future = past_future_split(ds, timestamp_column="timestamp", cutoff=30, horizon=10, min_context_length=20)

    assert len(past) == 100
    assert len(future) == 100
    for i in range(100):
        assert len(past[i]["target"]) == 30
        assert len(future[i]["target"]) == 10


def test_when_dataset_has_numpy_format_then_output_preserves_format():
    ds = Dataset.from_list([_make_ts_record("A", 10)])
    ds = ds.with_format("numpy")
    past, future = past_future_split(ds, timestamp_column="timestamp", cutoff=5, horizon=3, min_context_length=1)

    assert past.format["type"] == "numpy"
    assert future.format["type"] == "numpy"
    assert isinstance(past[0]["target"], np.ndarray)


# generate_univariate_targets_from_multivariate tests


def _make_multivariate_dataset(ids, lengths, target_cols, extra_cols=None):
    """Helper to build a multivariate time series dataset."""
    records = []
    for id_, length in zip(ids, lengths):
        record = {
            "id": id_,
            "timestamp": pd.date_range("2020", freq="D", periods=length),
        }
        for col in target_cols:
            record[col] = [float(hash((id_, col, t)) % 100) for t in range(length)]
        if extra_cols:
            for col, val in extra_cols.items():
                record[col] = val if not isinstance(val, list) else [val[0]] * length
        records.append(record)
    return Dataset.from_list(records)


def test_when_single_item_two_targets_then_expanded_correctly():
    ds = _make_multivariate_dataset(["A"], [5], ["X", "Y"])
    result = generate_univariate_targets_from_multivariate(
        ds, id_column="id", new_target_column="target", generate_univariate_targets_from=["X", "Y"]
    )
    assert len(result) == 2
    assert result[0]["id"] == "A_X"
    assert result[1]["id"] == "A_Y"
    assert list(result[0]["target"]) == list(ds[0]["X"])
    assert list(result[1]["target"]) == list(ds[0]["Y"])


def test_when_multiple_items_then_interleaved_correctly():
    ds = _make_multivariate_dataset(["A", "B"], [3, 3], ["X", "Y"])
    result = generate_univariate_targets_from_multivariate(
        ds, id_column="id", new_target_column="target", generate_univariate_targets_from=["X", "Y"]
    )
    assert len(result) == 4
    assert [result[i]["id"] for i in range(4)] == ["A_X", "A_Y", "B_X", "B_Y"]
    assert list(result[0]["target"]) == list(ds[0]["X"])
    assert list(result[1]["target"]) == list(ds[0]["Y"])
    assert list(result[2]["target"]) == list(ds[1]["X"])
    assert list(result[3]["target"]) == list(ds[1]["Y"])


def test_when_non_target_columns_present_then_they_are_repeated():
    ds = Dataset.from_list(
        [
            {
                "id": "A",
                "timestamp": pd.date_range("2020", freq="D", periods=3),
                "X": [1.0, 2.0, 3.0],
                "Y": [10.0, 20.0, 30.0],
                "covariate": [100.0, 200.0, 300.0],
            }
        ]
    )
    result = generate_univariate_targets_from_multivariate(
        ds, id_column="id", new_target_column="target", generate_univariate_targets_from=["X", "Y"]
    )
    assert len(result) == 2
    assert list(result[0]["covariate"]) == [100.0, 200.0, 300.0]
    assert list(result[1]["covariate"]) == [100.0, 200.0, 300.0]
    assert list(result[0]["timestamp"]) == list(result[1]["timestamp"])


def test_when_items_have_different_lengths_then_expanded_correctly():
    ds = _make_multivariate_dataset(["A", "B"], [3, 5], ["X", "Y"])
    result = generate_univariate_targets_from_multivariate(
        ds, id_column="id", new_target_column="target", generate_univariate_targets_from=["X", "Y"]
    )
    assert len(result) == 4
    assert len(result[0]["target"]) == 3  # A_X
    assert len(result[1]["target"]) == 3  # A_Y
    assert len(result[2]["target"]) == 5  # B_X
    assert len(result[3]["target"]) == 5  # B_Y


def test_when_dataset_has_indices_then_flattened_before_expansion():
    ds = _make_multivariate_dataset(["A", "B", "C"], [3, 3, 3], ["X", "Y"])
    ds_filtered = ds.select([0, 2])  # creates lazy _indices
    assert ds_filtered._indices is not None
    result = generate_univariate_targets_from_multivariate(
        ds_filtered, id_column="id", new_target_column="target", generate_univariate_targets_from=["X", "Y"]
    )
    assert len(result) == 4
    assert result[0]["id"] == "A_X"
    assert result[2]["id"] == "C_X"


# convert_long_df_to_hf_dataset tests


def _make_long_df(num_ids: int = 3, length: int = 5) -> pd.DataFrame:
    rows = []
    for i in range(num_ids):
        for t in range(length):
            rows.append(
                {
                    "item_id": f"id_{i}",
                    "timestamp": pd.Timestamp("2020-01-01") + pd.Timedelta(days=t),
                    "target": float(i * 100 + t),
                    "category": ["A", "B", "C"][i % 3],
                }
            )
    return pd.DataFrame(rows)


def test_when_long_df_provided_then_dynamic_and_static_columns_are_inferred():
    df = _make_long_df(num_ids=3, length=4)
    ds = fev.utils.convert_long_df_to_hf_dataset(df, id_column="item_id", static_columns=["category"])

    assert len(ds) == 3
    assert isinstance(ds.features["item_id"], datasets.Value)
    assert isinstance(ds.features["category"], datasets.Value)
    assert isinstance(ds.features["timestamp"], datasets.Sequence)
    assert isinstance(ds.features["target"], datasets.Sequence)
    assert ds[0]["item_id"] == "id_0"
    assert list(ds[0]["target"]) == [0.0, 1.0, 2.0, 3.0]
    assert ds[1]["item_id"] == "id_1"
    assert list(ds[1]["target"]) == [100.0, 101.0, 102.0, 103.0]


def test_when_long_df_is_unsorted_then_output_is_sorted_by_id_and_timestamp():
    df = _make_long_df(num_ids=2, length=3)
    df_shuffled = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    ds = fev.utils.convert_long_df_to_hf_dataset(df_shuffled, id_column="item_id")
    assert list(ds[0]["target"]) == [0.0, 1.0, 2.0]
    assert list(ds[1]["target"]) == [100.0, 101.0, 102.0]


def test_when_long_df_is_provided_then_validate_time_series_dataset_passes():
    df = _make_long_df(num_ids=3, length=10)
    ds = fev.utils.convert_long_df_to_hf_dataset(df, id_column="item_id", static_columns=["category"])
    fev.utils.validate_time_series_dataset(ds, id_column="item_id", timestamp_column="timestamp")


def test_when_long_df_provided_then_pyarrow_path_matches_old_groupby_output():
    # Sanity check: independently constructed reference matches the new pyarrow path.
    df = _make_long_df(num_ids=4, length=6)
    ds = fev.utils.convert_long_df_to_hf_dataset(df, id_column="item_id", static_columns=["category"])
    expected_per_id = {f"id_{i}": [float(i * 100 + t) for t in range(6)] for i in range(4)}
    for row in ds:
        assert list(row["target"]) == expected_per_id[row["item_id"]]

import datasets
import pandas as pd
import pytest

import fev


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

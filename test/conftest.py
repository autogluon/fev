import functools

import datasets
import pytest

from fev import Task


@pytest.fixture(scope="session", autouse=True)
def cache_hf_hub_datasets():
    """Load each HF Hub dataset once per test session.

    Every `datasets.load_dataset` call sends several metadata requests to the HF Hub, even when the data is already
    cached on disk. Tests load the same few Hub datasets many times, so reuse the loaded (immutable) dataset.
    Local files and calls with extra options are passed through unchanged.
    """
    original = datasets.load_dataset
    cached = functools.cache(original)

    def load_dataset(path, name=None, data_files=None, split=None, **kwargs):
        if data_files is None and not any(kwargs.values()):
            return cached(path, name, split=split)
        return original(path, name, data_files=data_files, split=split, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(datasets, "load_dataset", load_dataset)
        yield


@pytest.fixture(
    params=[
        {
            "dataset_path": "autogluon/chronos_datasets",
            "dataset_config": "monash_m1_yearly",
            "horizon": 8,
            "seasonality": 4,
            "num_windows": 2,
        },
        {
            "dataset_path": "autogluon/chronos_datasets",
            "dataset_config": "monash_m1_yearly",
            "horizon": 8,
            "seasonality": 1,
        },
    ]
)
def task_def(request):
    return Task(**request.param)


@pytest.fixture()
def dummy_task():
    return Task(
        dataset_path="autogluon/chronos_datasets",
        dataset_config="monash_m1_yearly",
        horizon=8,
    )

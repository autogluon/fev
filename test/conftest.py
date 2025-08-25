import pytest

from fev import Task


@pytest.fixture(
    params=[
        {
            "dataset_path": "autogluon/chronos_datasets",
            "dataset_config": "monash_m3_quarterly",
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

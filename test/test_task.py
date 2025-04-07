import datasets
import numpy as np
import pydantic
import pytest

import fev
import fev.task


def test_when_get_input_data_called_then_datasets_are_returned(task_def):
    train_data, future_data = task_def.get_input_data()
    assert isinstance(train_data, datasets.Dataset)
    assert isinstance(future_data, datasets.Dataset)


def test_when_get_input_data_called_then_datasets_contain_correct_columns(task_def):
    train_data, future_data = task_def.get_input_data()
    expected_train_columns = (
        [task_def.id_column, task_def.timestamp_column, task_def.target_column]
        + task_def.static_columns
        + task_def.dynamic_columns
    )
    assert set(expected_train_columns) == set(train_data.column_names)

    expected_future_columns = [task_def.id_column, task_def.timestamp_column] + [
        c for c in task_def.dynamic_columns if c not in task_def.past_dynamic_columns
    ]
    assert set(expected_future_columns) == set(future_data.column_names)


def test_when_list_of_config_provided_then_benchmark_can_be_loaded():
    task_configs = [
        {
            "dataset_path": "autogluon/chronos_datasets",
            "dataset_config": "monash_m1_yearly",
            "horizon": 8,
        },
        {
            "dataset_path": "autogluon/chronos_datasets",
            "dataset_config": "ercot",
            "horizon": 48,
            "seasonality": 24,
            "variants": [
                {"cutoff": "2021-01-01"},
                {"cutoff": "2021-02-01"},
            ],
        },
    ]
    benchmark = fev.Benchmark.from_list(task_configs)
    assert len(benchmark.tasks) == 3
    assert all(isinstance(task, fev.Task) for task in benchmark.tasks)


@pytest.mark.parametrize(
    "multiple_target_columns",
    [["price_mean"], ["price_mean", "distance_max", "distance_min"]],
)
def test_when_multiple_target_columns_used_then_one_instance_created_per_column(multiple_target_columns):
    task = fev.Task(
        dataset_path="autogluon/chronos_datasets",
        dataset_config="monash_rideshare",
        multiple_target_columns=multiple_target_columns,
    )
    original_ds = task._load_dataset()
    expanded_ds, _ = task.get_input_data()
    assert len(expanded_ds) == len(multiple_target_columns) * len(original_ds)
    assert len(expanded_ds.features) == len(original_ds.features) - len(multiple_target_columns) + 1
    assert len(np.unique(expanded_ds[task.id_column])) == len(expanded_ds)


def test_when_multiple_target_columns_set_to_all_used_then_all_columns_are_exploded():
    task = fev.Task(
        dataset_path="autogluon/chronos_datasets",
        dataset_config="monash_rideshare",
        multiple_target_columns=fev.task.MULTIPLE_TARGET_COLUMNS_ALL,
    )
    original_ds = task._load_dataset()
    num_sequence_columns = len(
        [
            col
            for col, feat in original_ds.features.items()
            if isinstance(feat, datasets.Sequence) and col != task.timestamp_column
        ]
    )
    expanded_ds, _ = task.get_input_data()
    assert len(expanded_ds) == num_sequence_columns * len(original_ds)
    assert len(expanded_ds.features) == len(original_ds.features) - num_sequence_columns + 1
    assert len(np.unique(expanded_ds[task.id_column])) == len(expanded_ds)


@pytest.mark.parametrize(
    "config",
    [
        {"variants": [], "num_rolling_windows": 3},
        {"variants": [], "rolling_step_size": 24},
        {"variants": [], "initial_cutoff": -48},
        {"num_rolling_windows": -1},
        {"num_rolling_windows": 2, "initial_cutoff": 48},
        {"num_rolling_windows": 2, "initial_cutoff": -48, "rolling_step_size": "24h"},
        {"num_rolling_windows": 2, "initial_cutoff": "2021-01-01", "rolling_step_size": 24},
        {"num_rolling_windows": 2, "initial_cutoff": "2021-01-01", "rolling_step_size": "-24h"},
        {"num_rolling_windows": 2, "initial_cutoff": "2021-01-01"},
        {"num_rolling_windows": 2, "rolling_step_size": "24h"},
    ],
)
def test_when_invalid_task_generator_config_provided_then_validation_error_is_raised(config):
    with pytest.raises(pydantic.ValidationError):
        fev.TaskGenerator(dataset_path="my_dataset", horizon=24, **config)


@pytest.mark.parametrize(
    "config, expected_cutoffs",
    [
        ({"num_rolling_windows": 3}, [-36, -24, -12]),
        ({"num_rolling_windows": 3, "initial_cutoff": -48}, [-48, -36, -24]),
        ({"num_rolling_windows": 3, "initial_cutoff": -48, "rolling_step_size": 4}, [-48, -44, -40]),
        ({"num_rolling_windows": 2, "rolling_step_size": 4}, [-24, -20]),
        (
            {"num_rolling_windows": 2, "initial_cutoff": "2024-06-01", "rolling_step_size": "4h"},
            ["2024-06-01T00:00:00", "2024-06-01T04:00:00"],
        ),
        (
            {"num_rolling_windows": 2, "initial_cutoff": "2024-06-01", "rolling_step_size": "1ME"},
            ["2024-06-01T00:00:00", "2024-06-30T00:00:00"],
        ),
    ],
)
def test_when_using_rolling_evaluation_then_tasks_are_generated_with_correct_offsets(config, expected_cutoffs):
    tasks = fev.TaskGenerator(dataset_path="my_dataset", horizon=12, **config).generate_tasks()
    assert [t.cutoff for t in tasks] == expected_cutoffs

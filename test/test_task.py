import datasets
import numpy as np
import pydantic
import pytest

import fev
import fev.task


def test_when_get_input_data_called_then_datasets_are_returned(task_def):
    past_data, future_data = task_def.get_input_data()
    assert isinstance(past_data, datasets.Dataset)
    assert isinstance(future_data, datasets.Dataset)


def test_when_get_input_data_called_then_datasets_contain_correct_columns(task_def):
    past_data, future_data = task_def.get_input_data()
    expected_train_columns = (
        [task_def.id_column, task_def.timestamp_column, task_def.target_column]
        + task_def.static_columns
        + task_def.dynamic_columns
    )
    assert set(expected_train_columns) == set(past_data.column_names)

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
    "generate_univariate_targets_from",
    [["price_mean"], ["price_mean", "distance_max", "distance_min"]],
)
def test_when_generate_univariate_targets_from_used_then_one_instance_created_per_column(
    generate_univariate_targets_from,
):
    task = fev.Task(
        dataset_path="autogluon/chronos_datasets",
        dataset_config="monash_rideshare",
        generate_univariate_targets_from=generate_univariate_targets_from,
    )
    original_ds = task._load_dataset()
    expanded_ds, _ = task.get_input_data()
    assert len(expanded_ds) == len(generate_univariate_targets_from) * len(original_ds)
    assert len(expanded_ds.features) == len(original_ds.features) - len(generate_univariate_targets_from) + 1
    assert len(np.unique(expanded_ds[task.id_column])) == len(expanded_ds)


def test_when_multiple_target_columns_set_to_all_used_then_all_columns_are_exploded():
    task = fev.Task(
        dataset_path="autogluon/chronos_datasets",
        dataset_config="monash_rideshare",
        generate_univariate_targets_from=fev.task.GENERATE_UNIVARIATE_TARGETS_FROM_ALL,
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


@pytest.mark.parametrize("target_column", [["OT"], ["OT", "LULL", "HULL"]])
def test_when_multivariate_task_is_created_then_data_contains_correct_columns(target_column):
    task = fev.Task(
        dataset_path="autogluon/chronos_datasets_extra",
        dataset_config="ETTh",
        target_column=target_column,
    )
    all_column_names = task._load_dataset().column_names
    past_data, future_data = task.get_input_data()
    assert set(past_data.column_names) == set(all_column_names)
    assert set(future_data.column_names) == set(all_column_names) - set(target_column)


@pytest.mark.parametrize("return_dict", [True, False])
def test_when_predictions_provided_as_dataset_dict_for_univariate_task_then_predictions_can_be_scored(return_dict):
    def naive_forecast_univariate(task: fev.Task) -> datasets.DatasetDict | list[dict]:
        past_data, future_data = task.get_input_data()
        predictions = []
        for ts in past_data:
            predictions.append({"predictions": [ts[task.target_column][-1] for _ in range(task.horizon)]})
        if return_dict:
            return datasets.DatasetDict({task.target_column: datasets.Dataset.from_list(predictions)})
        else:
            return predictions

    task = fev.Task(
        dataset_path="autogluon/chronos_datasets",
        dataset_config="monash_m1_yearly",
        eval_metric="MASE",
        extra_metrics=["WAPE"],
        horizon=4,
    )
    predictions = naive_forecast_univariate(task)
    summary = task.evaluation_summary(predictions, model_name="naive")
    for metric in ["MASE", "WAPE"]:
        assert np.isfinite(summary[metric])


@pytest.mark.parametrize("target_column", [["OT"], ["OT", "LULL", "HULL"]])
@pytest.mark.parametrize("return_dict", [True, False])
def test_when_multivariate_task_is_used_then_predictions_can_be_scored(target_column, return_dict):
    def naive_forecast_multivariate(task: fev.Task) -> datasets.DatasetDict | dict:
        past_data, future_data = task.get_input_data()
        predictions = {}
        for col in task.target_columns_list:
            predictions_for_column = []
            for ts in past_data:
                predictions_for_column.append({"predictions": [ts[col][-1] for _ in range(task.horizon)]})
            predictions[col] = predictions_for_column
        if return_dict:
            return predictions
        else:
            return datasets.DatasetDict({k: datasets.Dataset.from_list(v) for k, v in predictions.items()})

    task = fev.Task(
        dataset_path="autogluon/chronos_datasets_extra",
        dataset_config="ETTh",
        target_column=target_column,
        eval_metric="MASE",
        extra_metrics=["WAPE"],
        horizon=4,
    )

    predictions = naive_forecast_multivariate(task)
    summary = task.evaluation_summary(predictions, model_name="naive")
    for metric in ["MASE", "WAPE"]:
        assert np.isfinite(summary[metric])


@pytest.mark.parametrize("target_column", [["OT"], ["OT", "LULL", "HULL"]])
@pytest.mark.parametrize("return_type", ["dict", "list", "DatasetDict", "Dataset"])
@pytest.mark.parametrize("univariate_target_column", ["target", "CUSTOM_TARGET"])
def test_when_using_univariate_model_on_multivariate_task_via_adapters_then_predictions_can_be_scores(
    target_column, return_type, univariate_target_column
):
    def naive_forecast_univariate(task: fev.Task):
        past_data, future_data = fev.convert_input_data(
            task, adapter="datasets", as_univariate=True, univariate_target_column=univariate_target_column
        )
        predictions = []
        for ts in past_data:
            predictions.append({"predictions": [ts[univariate_target_column][-1] for _ in range(task.horizon)]})
        match return_type:
            case "list":
                return predictions
            case "dict":
                return {univariate_target_column: predictions}
            case "DatasetDict":
                return datasets.DatasetDict({univariate_target_column: datasets.Dataset.from_list(predictions)})
            case "Dataset":
                return datasets.Dataset.from_list(predictions)

    task = fev.Task(
        dataset_path="autogluon/chronos_datasets_extra",
        dataset_config="ETTh",
        target_column=target_column,
        eval_metric="MASE",
        extra_metrics=["WAPE"],
        horizon=4,
    )

    predictions = fev.utils.combine_univariate_predictions_to_multivariate(
        predictions=naive_forecast_univariate(task), target_columns_list=task.target_columns_list
    )
    summary = task.evaluation_summary(predictions, model_name="naive")
    for metric in ["MASE", "WAPE"]:
        assert np.isfinite(summary[metric])


@pytest.mark.parametrize(
    "horizon, cutoff, min_context_length, expected_num_items",
    [(8, None, 10, 419), (8, None, 1, 518), (30, None, 1, 31), (8, -20, 1, 406)],
)
def test_when_some_series_have_too_few_observations_then_they_get_filtered_out(
    horizon, cutoff, min_context_length, expected_num_items
):
    task = fev.Task(
        dataset_path="autogluon/chronos_datasets",
        dataset_config="monash_tourism_yearly",
        horizon=horizon,
        cutoff=cutoff,
        min_context_length=min_context_length,
    )
    assert len(task.get_input_data()[0]) == expected_num_items


@pytest.mark.parametrize(
    "horizon, cutoff, min_context_length",
    [(50, None, 1), (8, -50, 1), (8, None, 100), (8, "2020-01-01", 1), (8, "1903-05-01", 1)],
)
def test_when_all_series_have_too_few_observations_then_exception_is_raised(horizon, cutoff, min_context_length):
    task = fev.Task(
        dataset_path="autogluon/chronos_datasets",
        dataset_config="monash_tourism_yearly",
        horizon=horizon,
        cutoff=cutoff,
        min_context_length=min_context_length,
    )
    with pytest.raises(ValueError, match="All time series in the dataset are too short"):
        task.get_input_data()

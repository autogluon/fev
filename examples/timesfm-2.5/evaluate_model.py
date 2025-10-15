import time

import datasets
import numpy as np
import pandas as pd
import timesfm
import torch
from gluonts.transform import LastValueImputation

import fev

torch.set_float32_matmul_precision("high")

TIMESFM_MODEL_QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def batchify(lst: list, batch_size: int = 32):
    """Convert list into batches of desired size."""
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]


def predict_window(
    window: fev.EvaluationWindow,
    model: timesfm.TimesFM_2p5_200M_torch,
    quantile_levels: list[float],
    batch_size: int,
    return_mean: bool,
) -> tuple[datasets.DatasetDict, float]:
    quantile_to_index = {}
    # Ensure that 0.5 quantile is predicted
    task_quantiles = [0.5] + quantile_levels
    for q in task_quantiles:
        # We add 1 below to account for the first prediction which is the mean
        quantile_to_index[q] = int(np.argmin(np.abs(np.array(TIMESFM_MODEL_QUANTILES) - q))) + 1

    past_data, _ = fev.convert_input_data(window, adapter="datasets", as_univariate=True)
    past_data = past_data.with_format("numpy").cast_column("target", datasets.Sequence(datasets.Value("float32")))

    imputation = LastValueImputation()
    # We copy the array because datasets sometimes returns numpy arrays which are not writeable
    # See: https://github.com/huggingface/datasets/issues/616
    inputs = [imputation(t.copy()) for t in past_data["target"]]

    forecast_batches = []
    start_time = time.monotonic()
    for batch in batchify(inputs, batch_size=batch_size):
        mean_forecast, full_forecast = model.forecast(inputs=batch, horizon=window.horizon)
        if return_mean:
            forecast = {"predictions": mean_forecast}
        else:
            forecast = {"predictions": full_forecast[:, :, quantile_to_index[0.5]]}
        for q in quantile_levels:
            forecast[str(q)] = full_forecast[:, :, quantile_to_index[q]]
        forecast_batches.append(forecast)
    window_inference_time = time.monotonic() - start_time

    predictions = datasets.Dataset.from_dict(
        {
            k: np.concatenate([batch[k] for batch in forecast_batches], axis=0)
            for k in ["predictions"] + [str(q) for q in quantile_levels]
        }
    )

    return fev.utils.combine_univariate_predictions_to_multivariate(
        predictions, target_columns=window.target_columns
    ), window_inference_time


def predict_with_model(
    task: fev.Task,
    model_name: str = "google/timesfm-2.5-200m-pytorch",
    batch_size: int = 256,
    context_length: int = 16_000,
    per_core_batch_size: int = 64,
) -> tuple[list[datasets.DatasetDict], float, dict]:
    context_length = min(context_length, max([len(t) for t in task.load_full_dataset()[task.timestamp_column]]))
    print(f"Setting context_length={context_length}")

    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_name)
    model_hparams = dict(
        max_context=context_length,
        max_horizon=task.horizon,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=True,
        fix_quantile_crossing=True,
        per_core_batch_size=per_core_batch_size,
    )
    model.compile(timesfm.ForecastConfig(**model_hparams))

    inference_time = 0.0
    predictions_per_window = []
    for window in task.iter_windows():
        predictions, window_inference_time = predict_window(
            window,
            model=model,
            quantile_levels=task.quantile_levels,
            batch_size=batch_size,
            return_mean=task.eval_metric in ["MSE", "RMSE", "RMSSE"],
        )
        predictions_per_window.append(predictions)
        inference_time += window_inference_time

    extra_info = {"model_config": {"batch_size": batch_size, **model_hparams}}

    return predictions_per_window, inference_time, extra_info


if __name__ == "__main__":
    model_name = "google/timesfm-2.5-200m-pytorch"
    num_tasks = 2  # replace with `num_tasks = None` to run on all tasks

    benchmark = fev.Benchmark.from_yaml(
        "https://github.com/autogluon/fev/raw/refs/heads/main/benchmarks/fev_bench/tasks.yaml"
    )
    summaries = []
    for task in benchmark.tasks[:num_tasks]:
        predictions, inference_time, extra_info = predict_with_model(task, model_name=model_name)
        evaluation_summary = task.evaluation_summary(
            predictions,
            model_name=model_name,
            inference_time_s=inference_time,
            extra_info=extra_info,
        )
        print(evaluation_summary)
        summaries.append(evaluation_summary)

    # Show and save the results
    summary_df = pd.DataFrame(summaries)
    print(summary_df)
    summary_df.to_csv("timesfm-2.5.csv", index=False)

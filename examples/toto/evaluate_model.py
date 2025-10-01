import math
import time

import datasets
import numpy as np
import pandas as pd
import torch
from toto.data.util.dataset import MaskedTimeseries
from toto.inference.forecaster import TotoForecaster
from toto.model.toto import Toto
from tqdm.auto import tqdm

import fev

datasets.disable_progress_bars()


def batchify(lst: list, batch_size: int):
    """Convert list into batches of desired size."""
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]


def freq_to_seconds(freq: pd.offsets.BaseOffset | str) -> float:
    if isinstance(freq, str):
        freq = pd.tseries.frequencies.to_offset(freq)
    try:
        return freq.nanos / 1e9
    except ValueError:
        if isinstance(freq, pd.offsets.BusinessDay):
            return freq.n * 24 * 60 * 60
        elif isinstance(freq, pd.offsets.Week):
            return freq.n * 7 * 24 * 60 * 60
        elif isinstance(freq, pd.offsets.MonthBegin) or isinstance(freq, pd.offsets.MonthEnd):
            return 30 * 24 * 60 * 60
        elif isinstance(freq, pd.offsets.QuarterEnd) or isinstance(freq, pd.offsets.QuarterBegin):
            return 90 * 24 * 60 * 60
        elif isinstance(freq, pd.offsets.YearEnd) or isinstance(freq, pd.offsets.YearBegin):
            return 365.25 * 24 * 60 * 60
        else:
            raise ValueError(f"Cannot handle frequency of type {type(freq)}: {freq}")


def ffill(tensor: "torch.Tensor") -> "torch.Tensor":
    """Forward fill along the last axis"""

    assert tensor.ndim > 1
    nan_mask = torch.isnan(tensor)
    indices = torch.where(nan_mask, 0, torch.arange(tensor.shape[-1], device=tensor.device).expand_as(tensor))
    last_valid = torch.cummax(indices, dim=-1).values
    return torch.gather(tensor, dim=-1, index=last_valid)


def left_pad_and_stack_2d(tensors: list["torch.Tensor"]) -> "torch.Tensor":
    max_len = max(c.shape[-1] for c in tensors)
    padded = []
    for c in tensors:
        assert isinstance(c, torch.Tensor)
        assert c.ndim == 2
        padding = torch.full(size=(c.shape[0], max_len - c.shape[-1]), fill_value=torch.nan, device=c.device)
        padded.append(torch.concat((padding, c), dim=-1))
    return torch.stack(padded)


def predict_window(
    window: fev.EvaluationWindow,
    toto_forecaster: "TotoForecaster",
    quantile_levels: list[float],
    max_context_length: int,
    max_batch_variate_size: int,
    num_samples: int,
    samples_per_batch: int,
    time_delta_seconds: float,
    as_univariate: bool,
    return_mean: bool,
    device: str,
) -> tuple[datasets.DatasetDict, float]:
    if as_univariate:
        past_data, _ = fev.convert_input_data(window, adapter="datasets", as_univariate=True)
        target_columns = ["target"]
    else:
        past_data, _ = window.get_input_data()
        target_columns = window.target_columns

    past_data_features = past_data.features
    past_data_features.update({col: datasets.Sequence(datasets.Value("float32")) for col in target_columns})
    past_data = past_data.cast(past_data_features)

    num_variates = len(target_columns)
    inputs = [
        torch.tensor(np.stack(tuple(row.values()), axis=0), dtype=torch.float32)
        for row in past_data.select_columns(target_columns)
    ]

    forecast_batches = []
    batch_size = max(1, math.floor(max_batch_variate_size / num_variates))

    start_time = time.monotonic()
    for batch in tqdm(batchify(inputs, batch_size=batch_size), total=len(inputs) // batch_size):
        stacked_batch = left_pad_and_stack_2d(batch)
        stacked_batch = stacked_batch[..., -max_context_length:]
        stacked_batch = stacked_batch.to(device=device)
        # Impute missing values
        stacked_batch = ffill(stacked_batch)
        nan_mask = torch.isnan(stacked_batch)
        stacked_batch[nan_mask] = 0.0

        current_batch_size, _, context_length = stacked_batch.shape
        # each item in the batch is assigned a unique ID from [0, ..., current_batch_size - 1]
        id_mask = torch.arange(current_batch_size, dtype=torch.int, device=device)[:, None, None].repeat(
            1, num_variates, context_length
        )

        # FIXME: technically this should be a tensor of unix epochs but it is not used
        # by the current model (Datadog/Toto-Open-Base-1.0)
        timestamp_seconds = torch.zeros_like(stacked_batch, dtype=torch.int)
        time_interval_seconds = torch.full(
            (current_batch_size, 1), fill_value=time_delta_seconds, device=device, dtype=torch.int
        )

        masked_timeseries = MaskedTimeseries(
            series=stacked_batch,
            padding_mask=~nan_mask,
            id_mask=id_mask,
            timestamp_seconds=timestamp_seconds,
            time_interval_seconds=time_interval_seconds,
        )

        toto_forecast = toto_forecaster.forecast(
            masked_timeseries,
            prediction_length=window.horizon,
            num_samples=num_samples,
            samples_per_batch=samples_per_batch,
        )

        multivariate_forecast = {variate_name: {} for variate_name in target_columns}
        if return_mean:
            mean_forecast = toto_forecast.mean.cpu().numpy()
            for i, variate_name in enumerate(target_columns):
                multivariate_forecast[variate_name]["predictions"] = mean_forecast[:, i]
        else:
            median_forecast = toto_forecast.quantile(0.5).cpu().numpy()
            for i, variate_name in enumerate(target_columns):
                multivariate_forecast[variate_name]["predictions"] = median_forecast[:, i]

        for q in quantile_levels:
            quantile_forecast = toto_forecast.quantile(q).cpu().numpy()
            for i, variate_name in enumerate(target_columns):
                multivariate_forecast[variate_name][str(q)] = quantile_forecast[:, i]

        forecast_batches.append(multivariate_forecast)

    window_inference_time = time.monotonic() - start_time

    predictions_dict: dict = {}
    for variate_name in target_columns:
        predictions_dict[variate_name] = datasets.Dataset.from_dict(
            {
                k: np.concatenate([batch[variate_name][k] for batch in forecast_batches], axis=0)
                for k in ["predictions"] + [str(q) for q in quantile_levels]
            }
        )
    predictions = datasets.DatasetDict(predictions_dict)
    predictions.set_format("numpy")

    return predictions, window_inference_time


def predict_with_model(
    task: fev.Task,
    model_path: str = "Datadog/Toto-Open-Base-1.0",
    max_batch_variate_size: int = 24,
    num_samples: int = 256,
    samples_per_batch: int = 32,
    max_context_length: int = 4096,
    as_univariate: bool = False,
    compile_model: bool = False,
    device: str = "auto",
) -> tuple[list[datasets.DatasetDict], float, dict]:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    toto = Toto.from_pretrained(model_path)
    toto.to(device)
    if compile_model:
        toto.compile()
    toto_forecaster = TotoForecaster(toto.model)

    inference_time = 0.0
    predictions_per_window = []

    for window in task.iter_windows():
        predictions, window_inference_time = predict_window(
            window=window,
            toto_forecaster=toto_forecaster,
            quantile_levels=task.quantile_levels,
            max_context_length=max_context_length,
            max_batch_variate_size=max_batch_variate_size,
            num_samples=num_samples,
            samples_per_batch=samples_per_batch,
            return_mean=task.eval_metric in ["MSE", "RMSE", "RMSSE"],
            time_delta_seconds=freq_to_seconds(task.freq),
            as_univariate=as_univariate,
            device=device,
        )
        predictions_per_window.append(predictions)
        inference_time += window_inference_time

    extra_info = {
        "model_config": {
            "model_path": model_path,
            "max_batch_variate_size": max_batch_variate_size,
            "num_samples": num_samples,
            "samples_per_batch": samples_per_batch,
            "max_context_length": max_context_length,
            "as_univariate": as_univariate,
            "compile_model": compile_model,
            "device": device,
        }
    }

    return predictions_per_window, inference_time, extra_info


if __name__ == "__main__":
    model_path = "Datadog/Toto-Open-Base-1.0"
    num_tasks = 2  # replace with `num_tasks = None` to run on all tasks

    benchmark = fev.Benchmark.from_yaml("/fsx/ansarnd/repos/fev/benchmarks/fev_bench/tasks.yaml")
    summaries = []
    for task in benchmark.tasks[:num_tasks]:
        predictions, inference_time, extra_info = predict_with_model(task, model_path=model_path)
        evaluation_summary = task.evaluation_summary(
            predictions,
            model_name=model_path,
            inference_time_s=inference_time,
            extra_info=extra_info,
        )
        print(evaluation_summary)
        summaries.append(evaluation_summary)

    # Show and save the results
    summary_df = pd.DataFrame(summaries)
    print(summary_df)
    summary_df.to_csv("toto.csv", index=False)

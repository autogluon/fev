import time

import datasets
import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline, ForecastType
from tqdm.auto import tqdm

import fev

datasets.disable_progress_bars()


def batchify(lst: list, batch_size: int = 32):
    """Convert list into batches of desired size."""
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]


def predict_with_model(
    task: fev.Task,
    model_name: str = "amazon/chronos-bolt-small",
    batch_size: int = 32,
    device_map: str = "cuda",
    torch_dtype: str = torch.bfloat16,
    num_samples: int = 20,
    seed: int = 123,
) -> tuple[list[datasets.DatasetDict], float, dict]:
    pipeline = BaseChronosPipeline.from_pretrained(model_name, device_map=device_map, torch_dtype=torch_dtype)
    torch.manual_seed(seed)

    inference_time = 0.0
    quantile_levels = task.quantile_levels.copy()
    if 0.5 not in quantile_levels:
        quantile_levels.append(0.5)

    predictions_per_window = []
    for window in task.iter_windows():
        past_data, _ = fev.convert_input_data(window, adapter="datasets", as_univariate=True)
        past_data = past_data.with_format("torch").cast_column("target", datasets.Sequence(datasets.Value("float32")))

        quantiles_all = []
        mean_all = []

        start_time = time.monotonic()
        for batch in batchify(past_data["target"], batch_size=batch_size):
            quantiles, mean = pipeline.predict_quantiles(
                context=batch,
                prediction_length=task.horizon,
                limit_prediction_length=False,
                quantile_levels=quantile_levels,
            )

            quantiles_all.append(quantiles.numpy())
            mean_all.append(mean.numpy())
        inference_time += time.monotonic() - start_time

        quantiles_np = np.concatenate(quantiles_all, axis=0)  # [num_items, horizon, num_quantiles]
        mean_np = np.concatenate(mean_all, axis=0)  # [num_items, horizon]

        if task.eval_metric in ["MSE", "RMSE", "RMSSE"]:
            point_forecast = mean_np  # [num_items, horizon]
        else:
            # use median as the point forecast
            point_forecast = quantiles_np[:, :, quantile_levels.index(0.5)]  # [num_items, horizon]
        predictions_dict = {"predictions": point_forecast}

        for idx, level in enumerate(task.quantile_levels):
            predictions_dict[str(level)] = quantiles_np[:, :, idx]

        predictions_per_window.append(
            fev.utils.combine_univariate_predictions_to_multivariate(
                datasets.Dataset.from_dict(predictions_dict), target_columns=task.target_columns
            )
        )

    extra_info = {
        "model_config": {
            "model_name": model_name,
            "batch_size": batch_size,
            "device_map": device_map,
            "torch_dtype": str(torch_dtype),
            "num_samples": num_samples,
            "seed": seed,
        }
    }
    return predictions_per_window, inference_time, extra_info


if __name__ == "__main__":
    model_name = "amazon/chronos-bolt-small"
    num_tasks = 2  # replace with `num_tasks = None` to run on all tasks

    benchmark = fev.Benchmark.from_yaml(
        "https://raw.githubusercontent.com/autogluon/fev/refs/heads/main/benchmarks/chronos_zeroshot/tasks.yaml"
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
    summary_df.to_csv(f"{model_name}.csv", index=False)

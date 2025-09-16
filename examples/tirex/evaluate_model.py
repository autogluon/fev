import time

import datasets
import pandas as pd
import torch
from tirex import ForecastModel, load_model
from tirex.models.mixed_stack import skip_cuda

import fev

datasets.disable_progress_bars()


def predict_with_model(
    task: fev.Task,
    model_name: str = "NX-AI/TiRex",
    batch_size: int = 512,
    device_map: str = "cuda",
    compile: bool = False,
) -> tuple[list[datasets.DatasetDict], float, dict]:
    model: ForecastModel = load_model(model_name, device=device_map)
    if compile:
        model = torch.compile(model)

    inference_time = 0.0
    predictions_per_window = []
    for window in task.iter_windows(trust_remote_code=True):
        past_data, _ = fev.convert_input_data(window, adapter="datasets", as_univariate=True)
        past_data = past_data.with_format("torch").cast_column("target", datasets.Sequence(datasets.Value("float32")))
        loaded_targets = [t for t in past_data["target"]]

        start_time = time.monotonic()
        quantiles, means = model.forecast(
            loaded_targets, quantile_levels=task.quantile_levels, prediction_length=task.horizon, batch_size=batch_size
        )
        inference_time += time.monotonic() - start_time

        predictions_dict = {"predictions": means}
        for idx, level in enumerate(task.quantile_levels):
            predictions_dict[str(level)] = quantiles[:, :, idx]

        predictions_per_window.append(
            fev.combine_univariate_predictions_to_multivariate(
                datasets.Dataset.from_dict(predictions_dict), target_columns=task.target_columns
            )
        )

    extra_info = {
        "model_config": {
            "model_name": "tirex",
            "batch_size": batch_size,
            "device_map": device_map,
            "compile": compile,
            "cuda_kernel": not skip_cuda(),
        }
    }
    return predictions_per_window, inference_time, extra_info


if __name__ == "__main__":
    model_name = "NX-AI/TiRex"
    num_tasks = None  # replace with `num_tasks = None` to run on all tasks

    benchmark = fev.Benchmark.from_yaml(
        "https://raw.githubusercontent.com/autogluon/fev/refs/heads/main/benchmarks/chronos_zeroshot/tasks.yaml"
    )
    summaries = []
    for task in benchmark.tasks[:num_tasks]:
        predictions, inference_time, extra_info = predict_with_model(task, model_name=model_name)
        evaluation_summary = task.evaluation_summary(
            predictions,
            model_name="tirex",
            inference_time_s=inference_time,
            extra_info=extra_info,
        )
        print(evaluation_summary)
        summaries.append(evaluation_summary)

    # Show and save the results
    summary_df = pd.DataFrame(summaries)
    print(summary_df)
    summary_df.to_csv("tirex.csv", index=False)

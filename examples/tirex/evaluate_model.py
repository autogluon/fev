import time

import datasets
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

import fev
from tirex import load_model, ForecastModel
from tirex.models.mixed_stack import skip_cuda

datasets.disable_progress_bars()


def predict_with_model(
    task: fev.Task,
    model_name: str = "NX-AI/TiRex",
    batch_size: int = 512,
    device_map: str = "cuda",
    compile: bool = False,
) -> tuple[datasets.Dataset, float, dict]:
    model : ForecastModel = load_model(model_name, device=device_map)
    if compile:
        model = torch.compile(model)
    past_data, _ = task.get_input_data(trust_remote_code=True)
    target = past_data.with_format("torch").cast_column(
        task.target_column, datasets.Sequence(datasets.Value("float32"))
    )[task.target_column]
    loaded_targets = [t for t in target]

    quantile_levels = task.quantile_levels if task.quantile_levels is not None else []

    start_time = time.monotonic()
    quantiles, means = model.forecast(
        loaded_targets,
        quantile_levels=quantile_levels,
        prediction_length=task.horizon,
        batch_size=batch_size)
    inference_time = time.monotonic() - start_time

    predictions_dict = {"predictions": means}
    for idx, level in enumerate(quantile_levels):
        predictions_dict[str(level)] = quantiles[:, :, idx]

    predictions = datasets.Dataset.from_dict(predictions_dict)
    extra_info = {
        "model_config": {
            "model_name": "TiRex",
            "batch_size": batch_size,
            "device_map": device_map,
            "compile": compile,
            "cuda_kernel": not skip_cuda()
        }
    }
    return predictions, inference_time, extra_info


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
            model_name="TiRex",
            inference_time_s=inference_time,
            extra_info=extra_info,
        )
        print(evaluation_summary)
        summaries.append(evaluation_summary)    

    # Show and save the results
    summary_df = pd.DataFrame(summaries)
    print(summary_df)
    summary_df.to_csv(f"TiRex.csv", index=False)

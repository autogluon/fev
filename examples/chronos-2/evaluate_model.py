import chronos
import datasets
import pandas as pd
import torch
from chronos import BaseChronosPipeline

import fev

datasets.disable_progress_bars()


def predict_with_model(
    task: fev.Task,
    model_name: str = "s3://autogluon/chronos-2",
    batch_size: int = 100,
    device_map: str = "cuda",
    torch_dtype: torch.dtype = torch.float32,
    as_univariate: bool = False,
    predict_batches_jointly: bool = True,
    seed: int = 123,
) -> tuple[list[datasets.DatasetDict], float, dict]:
    pipeline = BaseChronosPipeline.from_pretrained(model_name, device_map=device_map, torch_dtype=torch_dtype)
    torch.manual_seed(seed)

    predictions_per_window, inference_time = pipeline.predict_fev(
        task, batch_size=batch_size, as_univariate=as_univariate, predict_batches_jointly=predict_batches_jointly
    )

    extra_info = {
        "framework_version": chronos.__version__,
        "model_config": {
            "model_name": model_name,
            "batch_size": batch_size,
            "device_map": device_map,
            "torch_dtype": str(torch_dtype),
            "as_univariate": as_univariate,
            "predict_batches_jointly": predict_batches_jointly,
            "seed": seed,
        },
    }
    return predictions_per_window, inference_time, extra_info


if __name__ == "__main__":
    model_name = "s3://autogluon/chronos-2"
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
    summary_df.to_csv("chronos-2.csv", index=False)

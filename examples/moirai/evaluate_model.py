import time

import datasets
import numpy as np
import pandas as pd
import torch
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

import fev

datasets.disable_progress_bars()


def predict_with_model(
    task: fev.Task,
    model_name: str = "Salesforce/moirai-1.1-R-large",
    context_length: int = 1024,
    batch_size: int = 128,
    num_samples: int = 100,
    device: str = "cuda",
    seed: int = 123,
) -> tuple[datasets.Dataset, float, dict]:
    _, prediction_dataset = fev.convert_input_data(task, "gluonts", trust_remote_code=True)

    torch.manual_seed(seed)
    model = MoiraiForecast(
        module=MoiraiModule.from_pretrained(model_name).to(device),
        prediction_length=task.horizon,
        context_length=context_length,
        num_samples=num_samples,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )
    predictor = model.create_predictor(batch_size=batch_size)

    start_time = time.monotonic()
    samples = np.stack([f.samples for f in predictor.predict(prediction_dataset)])
    inference_time = time.monotonic() - start_time

    if task.eval_metric in ["MSE", "RMSE", "RMSSE"]:
        point_forecast = np.mean(samples, axis=1)  # [num_items, horizon]
    else:
        point_forecast = np.median(samples, axis=1)  # [num_items, horizon]

    predictions_dict = {"predictions": point_forecast}
    if task.quantile_levels is not None:
        for q in task.quantile_levels:
            predictions_dict[str(q)] = np.quantile(samples, q=q, axis=1)  # [num_items, horizon]

    predictions = datasets.Dataset.from_dict(predictions_dict)
    extra_info = {
        "model_config": {
            "context_length": context_length,
            "model_name": model_name,
            "batch_size": batch_size,
            "num_samples": num_samples,
            "device": device,
            "seed": seed,
        }
    }

    return predictions, inference_time, extra_info


if __name__ == "__main__":
    model_name = "Salesforce/moirai-1.1-R-base"
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

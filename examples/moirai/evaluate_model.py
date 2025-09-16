import logging
import time
import warnings

import datasets
import numpy as np
import pandas as pd
import torch
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

import fev

datasets.disable_progress_bars()


def predict_with_model(
    task: fev.Task,
    model_name: str = "Salesforce/moirai-2.0-R-small",
    context_length: int = 500,
    batch_size: int = 128,
    device: str = "cpu",
    seed: int = 123,
) -> tuple[list[datasets.DatasetDict], float, dict]:
    torch.manual_seed(seed)
    # Disable GluonTS warnings when accessing forecast.mean
    gts_logger = logging.getLogger("gluonts")
    gts_logger.setLevel(100)

    model = Moirai2Forecast(
        module=Moirai2Module.from_pretrained(model_name).to(device),
        prediction_length=task.horizon,
        context_length=context_length,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )
    predictor = model.create_predictor(batch_size=batch_size)

    inference_time = 0.0
    predictions_per_window = []
    for window in task.iter_windows():
        _, prediction_dataset = fev.convert_input_data(window, adapter="gluonts", as_univariate=True)
        start_time = time.monotonic()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            forecasts = list(predictor.predict(prediction_dataset))
        inference_time += time.monotonic() - start_time

        predictions_dict = {"predictions": np.stack([f.mean for f in forecasts])}
        for q in task.quantile_levels:
            predictions_dict[str(q)] = np.stack([f.quantile(q) for f in forecasts])
        predictions_per_window.append(
            fev.utils.combine_univariate_predictions_to_multivariate(
                datasets.Dataset.from_dict(predictions_dict), target_columns=task.target_columns
            )
        )

    extra_info = {
        "model_config": {
            "context_length": context_length,
            "model_name": model_name,
            "batch_size": batch_size,
            "device": device,
            "seed": seed,
        }
    }

    return predictions_per_window, inference_time, extra_info


if __name__ == "__main__":
    model_name = "Salesforce/moirai-2.0-R-small"
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
    summary_df.to_csv("moirai-2.0.csv", index=False)

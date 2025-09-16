import time

import datasets
import numpy as np
import pandas as pd
from gluonts.model.seasonal_naive import SeasonalNaivePredictor

import fev


def predict_with_model(task: fev.Task) -> tuple[list[datasets.DatasetDict], float, dict]:
    predictor = SeasonalNaivePredictor(prediction_length=task.horizon, season_length=task.seasonality)

    inference_time = 0.0
    predictions_per_window = []
    for window in task.iter_windows(trust_remote_code=True):
        _, prediction_dataset = fev.convert_input_data(window, adapter="gluonts", as_univariate=True)
        start_time = time.monotonic()
        forecast = np.stack([f.samples for f in predictor.predict(prediction_dataset)]).squeeze(
            1
        )  # [num_items, horizon]
        inference_time += time.monotonic() - start_time

        predictions_dict = {"predictions": forecast}
        for q in task.quantile_levels:
            predictions_dict[str(q)] = forecast

        predictions_per_window.append(
            fev.combine_univariate_predictions_to_multivariate(
                datasets.Dataset.from_dict(predictions_dict), target_columns=task.target_columns
            )
        )

    return predictions_per_window, inference_time, {}


if __name__ == "__main__":
    model_name = "seasonal_naive"
    num_tasks = 2  # replace with `num_tasks = None` to run on all tasks

    benchmark = fev.Benchmark.from_yaml(
        "https://raw.githubusercontent.com/autogluon/fev/refs/heads/main/benchmarks/chronos_zeroshot/tasks.yaml"
    )
    summaries = []
    for task in benchmark.tasks[:num_tasks]:
        predictions, inference_time, extra_info = predict_with_model(task)
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

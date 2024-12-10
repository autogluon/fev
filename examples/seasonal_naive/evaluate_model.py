import time

import datasets
import numpy as np
from gluonts.model.seasonal_naive import SeasonalNaivePredictor

import fev


def predict_with_model(task: fev.Task) -> tuple[datasets.Dataset, float, dict]:
    _, prediction_dataset = fev.convert_input_data(task, "gluonts", trust_remote_code=True)

    predictor = SeasonalNaivePredictor(prediction_length=task.horizon, season_length=task.seasonality)

    start_time = time.monotonic()
    forecast = np.stack([f.samples for f in predictor.predict(prediction_dataset)]).squeeze(1)  # [num_items, horizon]
    inference_time = time.monotonic() - start_time

    predictions_dict = {"predictions": forecast}
    if task.quantile_levels is not None:
        for q in task.quantile_levels:
            predictions_dict[str(q)] = forecast

    predictions = datasets.Dataset.from_dict(predictions_dict)

    return predictions, inference_time, {}


if __name__ == "__main__":
    task = fev.Task(dataset_path="autogluon/chronos_datasets", dataset_config="monash_m1_yearly", horizon=8)
    predictions, inference_time, extra_info = predict_with_model(task)
    evaluation_summary = task.evaluation_summary(
        predictions,
        model_name="seasonal_naive",
        inference_time_s=inference_time,
    )
    print(evaluation_summary)

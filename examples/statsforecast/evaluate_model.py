import inspect
import time
import warnings
from typing import Type

import datasets
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoCES, AutoETS, AutoTheta, Naive, SeasonalNaive

import fev

datasets.disable_progress_bars()


model_name_to_class = {
    "naive": Naive,
    "seasonal_naive": SeasonalNaive,
    "auto_arima": AutoARIMA,
    "auto_ces": AutoCES,
    "auto_ets": AutoETS,
    "auto_theta": AutoTheta,
}


def filter_kwargs(cls: Type, kwargs: dict) -> dict:
    """Remove kwargs that are not expected by the given class object."""
    sig = inspect.signature(cls.__init__)
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return filtered_kwargs


def quantile_to_level(q: float) -> str:
    """Convert a numeric quantile value to the level suffix used by StatsForecast."""
    if q < 0.5:
        prefix = "-lo-"
        level = 100 - 200 * q
    else:
        prefix = "-hi-"
        level = 200 * q - 100
    return prefix + str(int(level))


def predict_with_model(
    task: fev.Task,
    model_name: str = "seasonal_naive",
    model_kwargs: dict | None = None,
    n_jobs: int = -1,
    max_context_length: int | None = 2500,
) -> tuple[datasets.Dataset, float, dict]:
    default_model_kwargs = {"season_length": task.seasonality}
    if model_kwargs is not None:
        default_model_kwargs.update(model_kwargs)

    model_cls = model_name_to_class[model_name]
    model = model_cls(**filter_kwargs(model_cls, default_model_kwargs))

    past_df, *_ = fev.convert_input_data(task, "nixtla", trust_remote_code=True)

    sf = StatsForecast(
        models=[model],
        freq=task.freq,
        n_jobs=n_jobs,
        fallback_model=SeasonalNaive(season_length=default_model_kwargs["season_length"]),
        verbose=True,
    )
    if task.quantile_levels is not None:
        levels = sorted(set([round(abs(q - 0.5) * 200) for q in task.quantile_levels]))
    else:
        levels = None

    if max_context_length is not None:
        past_df = past_df.groupby("unique_id").tail(max_context_length).reset_index(drop=True)

    start_time = time.monotonic()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forecast_df = sf.forecast(df=past_df, h=task.horizon, level=levels)
    inference_time = time.monotonic() - start_time

    # SeasonalNaive may return NaN forecast if len(y) < season_length; replace forecast with 0 in this case
    forecast_df = forecast_df.fillna(0.0)

    forecast_df["predictions"] = forecast_df[str(model)]
    if task.quantile_levels is not None:
        for q in task.quantile_levels:
            forecast_df[str(q)] = forecast_df[str(model) + quantile_to_level(q)]

    selected_columns = [fev.constants.PREDICTIONS]
    if task.quantile_levels is not None:
        selected_columns += [str(q) for q in task.quantile_levels]
    predictions_list = []
    for _, forecast in forecast_df.groupby("unique_id"):
        predictions_list.append(forecast[selected_columns].to_dict("list"))
    predictions = datasets.Dataset.from_list(predictions_list)

    return predictions, inference_time, {}


if __name__ == "__main__":
    task = fev.Task(dataset_path="autogluon/chronos_datasets", dataset_config="monash_m1_yearly", horizon=8)
    model_name = "auto_ets"
    predictions, inference_time, extra_info = predict_with_model(task, model_name=model_name)
    evaluation_summary = task.evaluation_summary(
        predictions,
        model_name=model_name,
        inference_time_s=inference_time,
    )
    print(evaluation_summary)

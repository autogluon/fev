import inspect
import os
import time
import warnings
from typing import Type

import datasets
import pandas as pd
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
    context_length: int | None = 2500,
) -> tuple[list[datasets.DatasetDict], float, dict]:
    default_model_kwargs = {"season_length": task.seasonality}
    if model_kwargs is not None:
        default_model_kwargs.update(model_kwargs)

    model_cls = model_name_to_class[model_name]
    model = model_cls(**filter_kwargs(model_cls, default_model_kwargs))

    sf = StatsForecast(
        models=[model],
        freq="D",  # we use a placeholder freq since we anyway ignore the forecast timestamps
        n_jobs=n_jobs,
        fallback_model=SeasonalNaive(season_length=default_model_kwargs["season_length"]),
        verbose=True,
    )
    levels = sorted(set([round(abs(q - 0.5) * 200) for q in task.quantile_levels]))

    inference_time = 0.0
    predictions_per_window = []
    for window in task.iter_windows(trust_remote_code=True):
        past_df, *_ = fev.convert_input_data(window, "nixtla", as_univariate=True)
        # Forward fill NaNs + zero-fill leading NaNs
        past_df = past_df.set_index("unique_id").groupby("unique_id").ffill().reset_index().fillna(0.0)
        if context_length is not None:
            past_df = past_df.groupby("unique_id").tail(context_length).reset_index(drop=True)

        start_time = time.monotonic()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            os.environ["PYTHONWARNINGS"] = "ignore"
            forecast_df = sf.forecast(df=past_df, h=task.horizon, level=levels)
        inference_time += time.monotonic() - start_time

        forecast_df["predictions"] = forecast_df[str(model)]
        for q in task.quantile_levels:
            forecast_df[str(q)] = forecast_df[str(model) + quantile_to_level(q)]

        selected_columns = ["predictions"] + [str(q) for q in task.quantile_levels]
        predictions_list = []
        for _, forecast in forecast_df.groupby("unique_id"):
            predictions_list.append(forecast[selected_columns].to_dict("list"))
        predictions_per_window.append(
            fev.combine_univariate_predictions_to_multivariate(
                datasets.Dataset.from_list(predictions_list), target_columns=task.target_columns
            )
        )

    extra_info = {"model_config": {"context_length": context_length, **default_model_kwargs}}

    return predictions_per_window, inference_time, extra_info


if __name__ == "__main__":
    model_name = "seasonal_naive"
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

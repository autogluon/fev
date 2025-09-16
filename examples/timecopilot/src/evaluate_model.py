import logging
import time
import warnings

import datasets
import pandas as pd
import typer
from timecopilot.models.ensembles.median import MedianEnsemble
from timecopilot.models.foundation.chronos import Chronos
from timecopilot.models.foundation.tirex import TiRex

import fev

app = typer.Typer()
logging.basicConfig(level=logging.INFO)
datasets.disable_progress_bars()


def predict_with_model(task: fev.Task) -> tuple[list[datasets.DatasetDict], float, dict]:
    forecaster = MedianEnsemble(
        models=[
            Chronos(
                repo_id="amazon/chronos-bolt-base",
                batch_size=256,
            ),
            TiRex(batch_size=256),
        ],
        alias="TimeCopilot",
    )

    renamer = {
        forecaster.alias: "predictions",
    }
    renamer.update({f"{forecaster.alias}-q-{int(100 * q)}": str(q) for q in task.quantile_levels})
    inference_time = 0.0
    predictions_per_window = []
    for window in task.iter_windows(trust_remote_code=True):
        past_df, *_ = fev.convert_input_data(window, "nixtla", as_univariate=True)
        # Forward fill NaNs + zero-fill leading NaNs
        past_df = past_df.set_index("unique_id").groupby("unique_id").ffill().reset_index().fillna(0.0)

        start_time = time.monotonic()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast_df = forecaster.forecast(
                df=past_df,
                h=task.horizon,
                quantiles=task.quantile_levels,
                freq=task.freq,
            )
        inference_time += time.monotonic() - start_time
        forecast_df = forecast_df.rename(columns=renamer)
        selected_columns = ["predictions"] + [str(q) for q in task.quantile_levels]
        predictions_list = []
        for _, forecast in forecast_df.groupby("unique_id"):
            predictions_list.append(forecast[selected_columns].to_dict("list"))
        predictions_per_window.append(
            fev.combine_univariate_predictions_to_multivariate(
                datasets.Dataset.from_list(predictions_list), target_columns=task.target_columns
            )
        )
    return predictions_per_window, inference_time, {}


def evaluate_task(task: fev.Task):
    predictions, inference_time, extra_info = predict_with_model(task)
    evaluation_summary = task.evaluation_summary(
        predictions,
        model_name="timecopilot",
        inference_time_s=inference_time,
        extra_info=extra_info,
    )
    print(evaluation_summary)
    return pd.DataFrame([evaluation_summary])


def tasks():
    benchmark = fev.Benchmark.from_yaml(
        "https://raw.githubusercontent.com/autogluon/fev/refs/heads/main/benchmarks/chronos_zeroshot/tasks.yaml"
    )
    return benchmark.tasks


@app.command()
def main(num_tasks: int | None = None):
    _tasks = tasks()[:num_tasks]
    logging.info(f"Evaluating {len(_tasks)} tasks")
    summaries = []
    for task in _tasks:
        evaluation_summary = evaluate_task(task)
        summaries.append(evaluation_summary)
    # Show and save the results
    summary_df = pd.concat(summaries)
    print(summary_df)
    summary_df.to_csv("timecopilot.csv", index=False)


if __name__ == "__main__":
    app()

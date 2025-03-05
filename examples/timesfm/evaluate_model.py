import time

import datasets
import numpy as np
import pandas as pd
import timesfm
from gluonts.transform import LastValueImputation
from timesfm.timesfm_torch import TimesFmTorch
from tqdm.auto import tqdm

import fev

datasets.disable_progress_bars()


def get_frequency_indicator(freqstr: str) -> int:
    base_freqstr = pd.tseries.frequencies.to_offset(freqstr).base.freqstr
    if base_freqstr[0] in ["Q", "Y"]:
        return 2
    elif base_freqstr[0] in ["W", "M"]:
        return 1
    else:
        return 0


def batchify(lst: list, batch_size: int = 32):
    """Convert list into batches of desired size."""
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]


VERSION_TO_HYPERPARAMETERS = {
    "google/timesfm-1.0-200m-pytorch": {
        "context_len": 512,
    },
    "google/timesfm-2.0-500m-pytorch": {
        "num_layers": 50,
        "use_positional_embedding": False,
        "context_len": 2048,
    },
}


def predict_with_model(
    task: fev.Task,
    model_name: str = "google/timesfm-2.0-500m-pytorch",
    backend: str = "gpu",
    batch_size: int = 256,
) -> tuple[datasets.Dataset, float, dict]:
    past_data, _ = task.get_input_data(trust_remote_code=True)
    target = past_data.with_format("numpy").cast_column(
        task.target_column, datasets.Sequence(datasets.Value("float32"))
    )[task.target_column]

    imputation = LastValueImputation()
    inputs = [imputation(t) for t in target]

    if model_name not in VERSION_TO_HYPERPARAMETERS:
        raise ValueError(f"model_name must be one of {list(VERSION_TO_HYPERPARAMETERS)} (got {model_name})")
    model_hparams = VERSION_TO_HYPERPARAMETERS[model_name]

    frequency_indicator = get_frequency_indicator(task.freq)
    tfm = TimesFmTorch(
        hparams=timesfm.TimesFmHparams(
            backend=backend,
            horizon_len=task.horizon,
            per_core_batch_size=32,
            **model_hparams,
        ),
        checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id=model_name),
    )

    quantile_to_index = {}
    # Ensure that 0.5 quantile is predicted
    task_quantiles = [0.5]
    if task.quantile_levels is not None:
        task_quantiles += task.quantile_levels
    for q in task_quantiles:
        # We add 1 below to account for the first prediction which is the mean
        quantile_to_index[q] = int(np.argmin(np.abs(np.array(tfm.quantiles) - q))) + 1

    forecast_batches = []
    start_time = time.monotonic()
    for batch in tqdm(batchify(inputs, batch_size=batch_size), total=len(inputs) // batch_size):
        mean_forecast, full_forecast = tfm.forecast(batch, freq=[frequency_indicator for _ in batch])

        if task.eval_metric in ["MSE", "RMSE", "RMSSE"]:
            forecast = {"predictions": mean_forecast}
        else:
            forecast = {"predictions": full_forecast[:, :, quantile_to_index[0.5]]}

        if task.quantile_levels is not None:
            for q in task.quantile_levels:
                forecast[str(q)] = full_forecast[:, :, quantile_to_index[q]]
        forecast_batches.append(forecast)

    inference_time = time.monotonic() - start_time
    predictions = datasets.Dataset.from_dict(
        {k: np.concatenate([batch[k] for batch in forecast_batches], axis=0) for k in task.predictions_schema.keys()}
    )
    extra_info = {
        "model_config": {
            "batch_size": batch_size,
            "backend": backend,
            "frequency_indicator": frequency_indicator,
            **model_hparams,
        }
    }

    return predictions, inference_time, extra_info


if __name__ == "__main__":
    model_name = "google/timesfm-2.0-500m-pytorch"
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
    summary_df.to_csv(f"{model_name.replace('/', '_')}.csv", index=False)

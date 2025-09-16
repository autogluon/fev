# fev
A lightweight library that makes it easy to benchmark time series forecasting models.

- Extensible: Easy to define your own forecasting tasks and benchmarks.
- Reproducible: Ensures that the results obtained by different users are comparable.
- Easy to use: Compatible with most popular forecasting libraries.
- Minimal dependencies: Just a thin wrapper on top of 🤗[`datasets`](https://huggingface.co/docs/datasets/en/index).

### How is `fev` different from other benchmarking tools?

Existing forecasting benchmarks usually fall into one of two categories:

- Standalone datasets without any supporting infrastructure. These provide no guarantees that the results obtained by different users are comparable. For example, changing the start date or duration of the forecast horizon totally changes the meaning of the scores.
- Bespoke end-to-end systems that combine models, datasets and forecasting tasks. Such packages usually come with lots of dependencies and assumptions, which makes extending or integrating these libraries into existing systems difficult.

`fev` aims for the middle ground - it provides the core benchmarking functionality without introducing unnecessary constraints or bloated dependencies. The library supports point & probabilistic forecasting, different types of covariates, as well as all popular forecasting metrics.

## 📝 Updates
- **2025-09-16**: The new version `0.6.0` contains major new functionality, [updated documentation](https://autogluon.github.io/fev/latest/), as well as some breaking changes to the `Task` API. Please check the [release notes](https://github.com/autogluon/fev/releases) for more details.

## ⚙️ Installation
```
pip install fev
```

## 🚀 Quickstart

Create a task from a dataset stored on Hugging Face Hub
```python
import fev

task = fev.Task(
    dataset_path="autogluon/chronos_datasets",
    dataset_config="m4_hourly",
    horizon=24,
)
```
Iterate over the rolling evaluation windows:
```python
for window in task.iter_windows():
    past_data, future_data = window.get_input_data()
```
- `past_data` contains the past data before the forecast horizon (item ID, past timestamps, target, all covariates).
- `future_data` contains future data that is known at prediction time (item ID, future timestamps, and known covariates)

Make predictions
```python
def naive_forecast(y: list, horizon: int) -> dict[str, list[float]]:
    # Make predictions for a single time series
    return {"predictions": [y[-1] for _ in range(horizon)]}

predictions_per_window = []
for window in task.iter_windows():
    past_data, future_data = window.get_input_data()
    predictions = [
        naive_forecast(ts[task.target_column], task.horizon) for ts in past_data
    ]
    predictions_per_window.append(predictions)
```
Get an evaluation summary
```python
task.evaluation_summary(predictions_per_window, model_name="naive")
# {'model_name': 'naive',
#  'dataset_path': 'autogluon/chronos_datasets',
#  'dataset_config': 'm4_hourly',
#  'horizon': 24,
#  'num_windows': 1,
#  'initial_cutoff': -24,
#  'window_step_size': 24,
#  'min_context_length': 1,
#  'max_context_length': None,
#  'seasonality': 1,
#  'eval_metric': 'MASE',
#  'extra_metrics': [],
#  'quantile_levels': None,
#  'id_column': 'id',
#  'timestamp_column': 'timestamp',
#  'target_column': 'target',
#  'generate_univariate_targets_from': None,
#  'past_dynamic_columns': [],
#  'excluded_columns': [],
#  'task_name': 'm4_hourly',
#  'test_error': 3.815112047601983,
#  'training_time_s': None,
#  'inference_time_s': None,
#  'dataset_fingerprint': '19e36bb78b718d8d',
#  'trained_on_this_dataset': False,
#  'fev_version': '0.6.0',
#  'MASE': 3.815112047601983}
```
The evaluation summary contains all information necessary to uniquely identify the forecasting task.

Multiple evaluation summaries produced by different models on different tasks can be aggregated into a single table.
```python
# Dataframes, dicts, JSON or CSV files supported
summaries = "https://raw.githubusercontent.com/autogluon/fev/refs/heads/main/benchmarks/example/results/results.csv"
fev.leaderboard(summaries)
# | model_name     |   skill_score |   win_rate | ... |
# |:---------------|--------------:|-----------:| ... |
# | auto_theta     |         0.126 |      0.667 | ... |
# | auto_arima     |         0.113 |      0.667 | ... |
# | auto_ets       |         0.049 |      0.444 | ... |
# | seasonal_naive |         0     |      0.222 | ... |
```

## 📚 Documentation
- Tutorials
    - [Quickstart](https://autogluon.github.io/fev/latest/tutorials/01-quickstart/): Define a task and evaluate a model.
    - [Datasets](https://autogluon.github.io/fev/latest/tutorials/02-dataset-format/): Use `fev` with your own datasets.
    - [Tasks & benchmarks](https://autogluon.github.io/fev/latest/tutorials/03-tasks-and-benchmarks/): Advanced features for defining tasks and benchmarks.
    - [Models](https://autogluon.github.io/fev/latest/tutorials/04-models/): Evaluate your models and submit results to the leaderboard.
- [API reference](https://autogluon.github.io/fev/latest/api/task/)

Examples of model implementations compatible with `fev` are available in [`examples/`](./examples/).


## 🏅 Leaderboards
We host leaderboards obtained using `fev` under https://huggingface.co/spaces/autogluon/fev-leaderboard.

Currently, the leaderboard includes the results from the Benchmark II introduced in [Chronos: Learning the Language of Time Series](https://arxiv.org/abs/2403.07815). We expect to extend this list in the future.

## 📈 Datasets
Repositories with datasets in format compatible with `fev`:
- [`chronos_datasets`](https://huggingface.co/datasets/autogluon/chronos_datasets)
- [`fev_datasets`](https://huggingface.co/datasets/autogluon/fev_datasets)

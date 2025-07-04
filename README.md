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

## Installation
```
pip install fev
```

## Quickstart

Create a task from a dataset stored on Hugging Face Hub
```python
import fev

task = fev.Task(
    dataset_path="autogluon/chronos_datasets",
    dataset_config="monash_kdd_cup_2018",
    horizon=12,
)
```
Load data available as input to the forecasting model
```python
past_data, future_data = task.get_input_data()
```
- `past_data` contains the past data before the forecast horizon (item ID, past timestamps, target, all covariates).
- `future_data` contains future data that is known at prediction time (item ID, future timestamps, and known covariates)

Make predictions
```python
def naive_forecast(y: list, horizon: int) -> list:
    return [y[-1] for _ in range(horizon)]

predictions = []
for ts in past_data:
    predictions.append(
        {"predictions": naive_forecast(y=ts[task.target_column], horizon=task.horizon)}
    )
```
Get an evaluation summary
```python
task.evaluation_summary(predictions, model_name="naive")
# {'model_name': 'naive',
#  'dataset_name': 'chronos_datasets_monash_kdd_cup_2018',
#  'dataset_path': 'autogluon/chronos_datasets',
#  'dataset_config': 'monash_kdd_cup_2018',
#  'horizon': 12,
#  'cutoff': -12,
#  'lead_time': 1,
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
#  'test_error': 3.3784518866750513,
#  'training_time_s': None,
#  'inference_time_s': None,
#  'dataset_fingerprint': 'a22d13d4c1e8641c',
#  'trained_on_this_dataset': False,
#  'fev_version': '0.5.0',
#  'MASE': 3.3784518866750513}
```
The evaluation summary contains all information necessary to uniquely identify the forecasting task.

Multiple evaluation summaries produced by different models on different tasks can be aggregated into a single table.
```python
# Dataframes, dicts, JSON or CSV files supported
summaries = "https://raw.githubusercontent.com/autogluon/fev/refs/heads/main/benchmarks/example/results/results.csv"
fev.leaderboard(summaries)
# | model_name     |   gmean_relative_error |   avg_rank |   avg_inference_time_s |   ... |
# |:---------------|-----------------------:|-----------:|-----------------------:|------:|
# | auto_theta     |                  0.874 |      2     |                  5.501 |   ... |
# | auto_arima     |                  0.887 |      2     |                 21.799 |   ... |
# | auto_ets       |                  0.951 |      2.667 |                  0.737 |   ... |
# | seasonal_naive |                  1     |      3.333 |                  0.004 |   ... |
```

## Tutorials
- [Quickstart](./docs/01-quickstart.ipynb): Define a task and evaluate a model.
- [Datasets](./docs/02-dataset-format.ipynb): Use `fev` with your own datasets.
- [Tasks & benchmarks](./docs/03-tasks-and-benchmarks.ipynb): Advanced features for defining tasks and benchmarks.
- [Models](./docs/04-models.ipynb): Evaluate your models and submit results to the leaderboard.

Examples of model implementations compatible with `fev` are available in [`examples/`](./examples/).


## Leaderboards
We host leaderboards obtained using `fev` under https://huggingface.co/spaces/autogluon/fev-leaderboard.

Currently, the leaderboard includes the results from the Benchmark II introduced in [Chronos: Learning the Language of Time Series](https://arxiv.org/abs/2403.07815). We expect to extend this list in the future.

## Datasets
Repositories with datasets in format compatible with `fev`:
- [`chronos_datasets`](https://huggingface.co/datasets/autogluon/chronos_datasets)
- [`fev_datasets`](https://huggingface.co/datasets/autogluon/fev_datasets)

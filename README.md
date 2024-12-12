# fev: Forecast evaluation library
`fev` adds a thin wrapper on top of the Hugging Face (HF) [`datasets`](https://huggingface.co/docs/datasets/en/index) library, resulting in a lightweight but fully functional benchmarking solution.

Specifically, this package makes it easy to
- define time series forecasting tasks
- load time series data and generate train-test splits
- evaluate model predictions

`fev` supports point & probabilistic forecasting, different types of covariates, as well as all popular forecasting metrics.

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
#  'dataset_fingerprint': '8a50d3417859652b',
#  'dataset_path': 'autogluon/chronos_datasets',
#  'dataset_config': 'monash_kdd_cup_2018',
#  'horizon': 12,
#  'cutoff': -12,
#  'lead_time': 1,
#  'min_ts_length': 13,
#  'max_context_length': None,
#  'seasonality': 1,
#  'eval_metric': 'MASE',
#  'extra_metrics': [],
#  'quantile_levels': None,
#  'id_column': 'id',
#  'timestamp_column': 'timestamp',
#  'target_column': 'target',
#  'multiple_target_columns': None,
#  'past_dynamic_columns': [],
#  'excluded_columns': [],
#  'test_error': 3.3784518,
#  'training_time_s': None,
#  'inference_time_s': None,
#  'fev_version': '0.2.0',
#  'MASE': 3.3784518}
```
The evaluation summary contains all information necessary to uniquely identify the forecasting task.


## Tutorials
- Quick start tutorial: [docs/tutorials/quickstart.ipynb](./docs/tutorials/quickstart.ipynb).
- An in-depth walkthrough of the library: [docs/tutorials/in-depth.ipynb](./docs/tutorials/in-depth.ipynb).

Examples of model implementations compatible with `fev` are available in [`examples/`](./examples/).

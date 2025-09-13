# fev: Forecast evaluation library

`fev` is a lightweight library that makes it easy to benchmark time series forecasting models.

- **Extensible**: Easy to define your own forecasting tasks and benchmarks.
- **Reproducible**: Ensures that the results obtained by different users are comparable.
- **Easy to use**: Compatible with most popular forecasting libraries.
- **Minimal dependencies**: Just a thin wrapper on top of 🤗[`datasets`](https://huggingface.co/docs/datasets/en/index).

## Installation
```
pip install fev
```

## Quickstart

```python
import fev

# Create a forecasting task
task = fev.Task(
   dataset_path="autogluon/chronos_datasets",
   dataset_config="m4_hourly",
   horizon=24,
)

# Evaluate your model
predictions_per_window = []
for window in task.iter_windows():
   past_data, future_data = window.get_input_data()
   # Make predictions
   predictions_per_window.append(model.predict(past_data, future_data))

# Get reproducible evaluation summary with all task details & metrics
summary = task.evaluation_summary(predictions_per_window, "my_model")
```

## Tutorials
- 🚀 **[Quickstart](tutorials/01-quickstart.ipynb)** - Get started with your first forecasting task
- 📊 **[Dataset Format](tutorials/02-dataset-format.ipynb)** - Learn how to use your own datasets
- ⚙️ **[Tasks & Benchmarks](tutorials/03-tasks-and-benchmarks.ipynb)** - Advanced task configuration
- 🤖 **[Models](tutorials/04-models.ipynb)** - Integrate your forecasting models


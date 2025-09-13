# Metrics

For the precise mathematical definitions of metrics, see [AutoGluon documentation](https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-metrics.html).

**Note:** Currently, multivariate metrics are computed by first computing the univariate metric on each target column
and then averaging the results, similar to the following:
```python
metric_value = np.mean(
    [metric.compute_metric(test_data[col], predictions[col])
    for col in task.target_columns]
)
```
For some metrics like WAPE, this leads to results that are different from first concatenating all target columns into a
single array and computing the metric on it.

::: fev.metrics
    options:
      filters:
        - "!Metric"
        - "!^_"
      show_root_toc_entry: false

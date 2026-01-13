from typing import Any, Callable, Type

import datasets
import numpy as np

from fev.constants import PREDICTIONS

MetricConfig = str | dict[str, Any]


class Metric:
    """Base class for all metrics."""

    needs_quantiles: bool = False

    @property
    def name(self) -> str:
        """Name of the metric."""
        return self.__class__.__name__

    @staticmethod
    def _safemean(arr: np.ndarray) -> float:
        """Compute mean of an array, ignoring NaN, Inf, and -Inf values."""
        return float(np.mean(arr[np.isfinite(arr)]))

    def compute(
        self,
        *,
        test_data: datasets.Dataset,
        predictions: datasets.Dataset,
        past_data: datasets.Dataset,
        seasonality: int,
        quantile_levels: list[float],
        target_column: str = "target",
    ) -> float:
        raise NotImplementedError


def get_metric(metric: MetricConfig) -> Metric:
    """Get a metric class by name or configuration."""
    metric_name = metric if isinstance(metric, str) else metric["name"]
    try:
        metric_type = AVAILABLE_METRICS[metric_name.upper()]
    except KeyError:
        raise ValueError(
            f"Evaluation metric '{metric_name}' is not available. Available metrics: {sorted(AVAILABLE_METRICS)}"
        )

    if isinstance(metric, str):
        return metric_type()
    elif isinstance(metric, dict):
        return metric_type(**{k: v for k, v in metric.items() if k != "name"})
    else:
        raise ValueError(f"Invalid metric configuration: {metric}")


class MAE(Metric):
    """Mean absolute error."""

    def compute(
        self,
        *,
        test_data: datasets.Dataset,
        predictions: datasets.Dataset,
        past_data: datasets.Dataset,
        seasonality: int,
        quantile_levels: list[float],
        target_column: str = "target",
    ):
        y_test = np.array(test_data[target_column])
        y_pred = np.array(predictions[PREDICTIONS])
        return np.nanmean(np.abs(y_test - y_pred))


class WAPE(Metric):
    """Weighted absolute percentage error."""

    def __init__(self, epsilon: float = 0.0) -> None:
        self.epsilon = epsilon

    def compute(
        self,
        *,
        test_data: datasets.Dataset,
        predictions: datasets.Dataset,
        past_data: datasets.Dataset,
        seasonality: int,
        quantile_levels: list[float],
        target_column: str = "target",
    ):
        y_test = np.array(test_data[target_column])
        y_pred = np.array(predictions[PREDICTIONS])

        return np.nanmean(np.abs(y_test - y_pred)) / max(self.epsilon, np.nanmean(np.abs(y_test)))


class MASE(Metric):
    """Mean absolute scaled error."""

    def __init__(self, epsilon: float = 0.0) -> None:
        self.epsilon = epsilon

    def compute(
        self,
        *,
        test_data: datasets.Dataset,
        predictions: datasets.Dataset,
        past_data: datasets.Dataset,
        seasonality: int,
        quantile_levels: list[float],
        target_column: str = "target",
    ):
        y_test = np.array(test_data[target_column])
        y_pred = np.array(predictions[PREDICTIONS])

        seasonal_error = _abs_seasonal_error_per_item(
            past_data=past_data, seasonality=seasonality, target_column=target_column
        )
        seasonal_error = np.clip(seasonal_error, self.epsilon, None)
        return self._safemean(np.abs(y_test - y_pred) / seasonal_error[:, None])


class RMSE(Metric):
    """Root mean squared error."""

    def compute(
        self,
        *,
        test_data: datasets.Dataset,
        predictions: datasets.Dataset,
        past_data: datasets.Dataset,
        seasonality: int,
        quantile_levels: list[float],
        target_column: str = "target",
    ):
        y_test = np.array(test_data[target_column])
        y_pred = np.array(predictions[PREDICTIONS])
        return np.sqrt(np.nanmean((y_test - y_pred) ** 2))


class RMSSE(Metric):
    """Root mean squared scaled error."""

    def __init__(self, epsilon: float = 0.0) -> None:
        self.epsilon = epsilon

    def compute(
        self,
        *,
        test_data: datasets.Dataset,
        predictions: datasets.Dataset,
        past_data: datasets.Dataset,
        seasonality: int,
        quantile_levels: list[float],
        target_column: str = "target",
    ):
        y_test = np.array(test_data[target_column])
        y_pred = np.array(predictions[PREDICTIONS])
        seasonal_error = _squared_seasonal_error_per_item(
            past_data, seasonality=seasonality, target_column=target_column
        )
        seasonal_error = np.clip(seasonal_error, self.epsilon, None)
        return np.sqrt(self._safemean((y_test - y_pred) ** 2 / seasonal_error[:, None]))


class MSE(Metric):
    """Mean squared error."""

    def compute(
        self,
        *,
        test_data: datasets.Dataset,
        predictions: datasets.Dataset,
        past_data: datasets.Dataset,
        seasonality: int,
        quantile_levels: list[float],
        target_column: str = "target",
    ):
        y_test = np.array(test_data[target_column])
        y_pred = np.array(predictions[PREDICTIONS])
        return np.nanmean((y_test - y_pred) ** 2)


class RMSLE(Metric):
    """Root mean squared logarithmic error."""

    def compute(
        self,
        *,
        test_data: datasets.Dataset,
        predictions: datasets.Dataset,
        past_data: datasets.Dataset,
        seasonality: int,
        quantile_levels: list[float],
        target_column: str = "target",
    ):
        y_test = np.array(test_data[target_column])
        y_pred = np.array(predictions[PREDICTIONS])
        return np.sqrt(np.nanmean((np.log1p(y_test) - np.log1p(y_pred)) ** 2))


class MAPE(Metric):
    """Mean absolute percentage error."""

    def compute(
        self,
        *,
        test_data: datasets.Dataset,
        predictions: datasets.Dataset,
        past_data: datasets.Dataset,
        seasonality: int,
        quantile_levels: list[float],
        target_column: str = "target",
    ):
        y_test = np.array(test_data[target_column])
        y_pred = np.array(predictions[PREDICTIONS])
        ratio = np.abs(y_test - y_pred) / np.abs(y_test)
        return self._safemean(ratio)


class SMAPE(Metric):
    """Symmetric mean absolute percentage error."""

    def compute(
        self,
        *,
        test_data: datasets.Dataset,
        predictions: datasets.Dataset,
        past_data: datasets.Dataset,
        seasonality: int,
        quantile_levels: list[float],
        target_column: str = "target",
    ):
        y_test = np.array(test_data[target_column])
        y_pred = np.array(predictions[PREDICTIONS])
        return self._safemean(2 * np.abs(y_test - y_pred) / (np.abs(y_test) + np.abs(y_pred)))


class MQL(Metric):
    """Mean quantile loss."""

    needs_quantiles: bool = True

    def compute(
        self,
        *,
        test_data: datasets.Dataset,
        predictions: datasets.Dataset,
        past_data: datasets.Dataset,
        seasonality: int,
        quantile_levels: list[float],
        target_column: str = "target",
    ):
        if quantile_levels is None or len(quantile_levels) == 0:
            raise ValueError(f"{self.__class__.__name__} cannot be computed if quantile_levels is None")
        ql = _quantile_loss(
            test_data=test_data,
            predictions=predictions,
            quantile_levels=quantile_levels,
            target_column=target_column,
        )
        return np.nanmean(ql)


class SQL(Metric):
    """Scaled quantile loss."""

    needs_quantiles: bool = True

    def __init__(self, epsilon: float = 0.0) -> None:
        self.epsilon = epsilon

    def compute(
        self,
        *,
        test_data: datasets.Dataset,
        predictions: datasets.Dataset,
        past_data: datasets.Dataset,
        seasonality: int,
        quantile_levels: list[float],
        target_column: str = "target",
    ):
        ql = _quantile_loss(
            test_data=test_data,
            predictions=predictions,
            quantile_levels=quantile_levels,
            target_column=target_column,
        )
        ql_per_time_step = np.nanmean(ql, axis=2)  # [num_items, horizon]
        seasonal_error = _abs_seasonal_error_per_item(
            past_data=past_data, seasonality=seasonality, target_column=target_column
        )
        seasonal_error = np.clip(seasonal_error, self.epsilon, None)
        return self._safemean(ql_per_time_step / seasonal_error[:, None])


class WQL(Metric):
    """Weighted quantile loss."""

    needs_quantiles: bool = True

    def __init__(self, epsilon: float = 0.0) -> None:
        self.epsilon = epsilon

    def compute(
        self,
        *,
        test_data: datasets.Dataset,
        predictions: datasets.Dataset,
        past_data: datasets.Dataset,
        seasonality: int,
        quantile_levels: list[float],
        target_column: str = "target",
    ):
        ql = _quantile_loss(
            test_data=test_data,
            predictions=predictions,
            quantile_levels=quantile_levels,
            target_column=target_column,
        )
        return np.nanmean(ql) / max(self.epsilon, np.nanmean(np.abs(np.array(test_data[target_column]))))


def _seasonal_error_per_item(
    arrays: list[np.ndarray],
    seasonality: int,
    aggregate_fn: Callable,
    nan_fill_value: float = 1.0,
) -> np.ndarray:
    """Compute seasonal error for each time series using vectorized operations.

    Uses bincount with weights to efficiently compute per-series aggregations.
    """
    num_series = len(arrays)
    if num_series == 0:
        return np.array([], dtype="float64")

    lengths = np.array([a.size for a in arrays], dtype=np.int64)
    num_diffs_per_series = np.maximum(lengths - seasonality, 0)

    if num_diffs_per_series.sum() == 0:
        return np.full(num_series, nan_fill_value, dtype="float64")

    flat = np.concatenate(arrays)
    series_starts = np.concatenate([[0], np.cumsum(lengths[:-1])])

    # Build indices for all (t, t-seasonality) pairs across all series
    total_diffs = int(num_diffs_per_series.sum())
    series_ids = np.repeat(np.arange(num_series, dtype=np.int64), num_diffs_per_series)
    diff_offsets = np.arange(total_diffs) - np.repeat(
        np.cumsum(num_diffs_per_series) - num_diffs_per_series, num_diffs_per_series
    )

    idx_current = series_starts[series_ids] + seasonality + diff_offsets
    idx_lagged = idx_current - seasonality

    diffs = flat[idx_current] - flat[idx_lagged]
    errors = aggregate_fn(diffs)

    # Compute per-series nanmean via bincount
    valid = ~np.isnan(errors)
    sums = np.bincount(series_ids, weights=np.where(valid, errors, 0.0), minlength=num_series)
    counts = np.bincount(series_ids, weights=valid.astype("float64"), minlength=num_series)

    result = np.full(num_series, nan_fill_value, dtype="float64")
    np.divide(sums, counts, out=result, where=counts > 0)
    return result


def _abs_seasonal_error_per_item(past_data: datasets.Dataset, seasonality: int, target_column: str) -> np.ndarray:
    """Compute mean absolute seasonal error for each time series in past_data."""
    arrays = past_data.with_format("numpy")[target_column]
    return _seasonal_error_per_item(arrays, seasonality, aggregate_fn=np.abs)


def _squared_seasonal_error_per_item(past_data: datasets.Dataset, seasonality: int, target_column: str) -> np.ndarray:
    """Compute mean squared seasonal error for each time series in past_data."""
    arrays = past_data.with_format("numpy")[target_column]
    return _seasonal_error_per_item(arrays, seasonality, aggregate_fn=np.square)


def _quantile_loss(
    *,
    test_data: datasets.Dataset,
    predictions: datasets.Dataset,
    quantile_levels: list[float],
    target_column: str,
):
    """Compute quantile loss for each observation"""
    pred_per_quantile = []
    for q in quantile_levels:
        pred_per_quantile.append(np.array(predictions[str(q)]))
    q_pred = np.stack(pred_per_quantile, axis=-1)  # [num_series, horizon, len(quantile_levels)]
    y_test = np.array(test_data[target_column])[..., None]  # [num_series, horizon, 1]
    assert y_test.shape[:-1] == q_pred.shape[:-1]
    return 2 * np.abs((y_test - q_pred) * ((y_test <= q_pred) - np.array(quantile_levels)))


AVAILABLE_METRICS: dict[str, Type[Metric]] = {
    # Median estimation
    "MAE": MAE,
    "WAPE": WAPE,
    "MASE": MASE,
    # Mean estimation
    "MSE": MSE,
    "RMSE": RMSE,
    "RMSSE": RMSSE,
    # Logarithmic errors
    "RMSLE": RMSLE,
    # Percentage errors
    "MAPE": MAPE,
    "SMAPE": SMAPE,
    # Quantile loss
    "MQL": MQL,
    "WQL": WQL,
    "SQL": SQL,
}

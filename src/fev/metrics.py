from typing import Any, Callable, Type

import numpy as np

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
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_past: np.ndarray,
        y_past_indptr: np.ndarray,
        q_pred: np.ndarray,
        seasonality: int,
        quantile_levels: list[float],
    ) -> float:
        """Compute the metric score.

        Parameters
        ----------
        y_true : np.ndarray [N, D, H]
            Ground truth values (N items, D target dims, H horizon steps).
        y_pred : np.ndarray [N, D, H]
            Point forecast predictions.
        y_past : np.ndarray [total_T, D]
            Concatenated past observations for all items. Use y_past_indptr to
            slice per item: item i has y_past[indptr[i]:indptr[i+1], :].
        y_past_indptr : np.ndarray [N+1]
            CSR-style index pointer into y_past.
        q_pred : np.ndarray [N, D, H, Q]
            Quantile predictions. Empty (Q=0) if no quantiles were requested.
        seasonality : int
            Seasonal period used for scaled error metrics.
        quantile_levels : list[float]
            Quantile levels corresponding to q_pred's last axis.
        """
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
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_past: np.ndarray,
        y_past_indptr: np.ndarray,
        q_pred: np.ndarray,
        seasonality: int,
        quantile_levels: list[float],
    ) -> float:
        per_dim = np.nanmean(np.abs(y_true - y_pred), axis=(0, 2))  # [D]
        return float(np.mean(per_dim))


class WAPE(Metric):
    """Weighted absolute percentage error."""

    def __init__(self, epsilon: float = 0.0) -> None:
        self.epsilon = epsilon

    def compute(
        self,
        *,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_past: np.ndarray,
        y_past_indptr: np.ndarray,
        q_pred: np.ndarray,
        seasonality: int,
        quantile_levels: list[float],
    ) -> float:
        abs_err_per_dim = np.nanmean(np.abs(y_true - y_pred), axis=(0, 2))  # [D]
        abs_true_per_dim = np.nanmean(np.abs(y_true), axis=(0, 2))  # [D]
        per_dim = abs_err_per_dim / np.maximum(abs_true_per_dim, self.epsilon)
        return float(np.mean(per_dim))


class MASE(Metric):
    """Mean absolute scaled error.

    Warning:
        Items with undefined in-sample seasonal error (e.g., history shorter than `seasonality`,
        all-NaN history, or zero seasonal error) are excluded from aggregation.
    """

    def __init__(self, epsilon: float = 0.0) -> None:
        self.epsilon = epsilon

    def compute(
        self,
        *,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_past: np.ndarray,
        y_past_indptr: np.ndarray,
        q_pred: np.ndarray,
        seasonality: int,
        quantile_levels: list[float],
    ) -> float:
        seasonal_error = _abs_seasonal_error(y_past=y_past, indptr=y_past_indptr, seasonality=seasonality)  # [N, D]
        seasonal_error = np.clip(seasonal_error, self.epsilon, None)
        # Per-dim MASE: safemean over [N, H] for each dim d, then average across D
        scaled = np.abs(y_true - y_pred) / seasonal_error[:, :, None]  # [N, D, H]
        per_dim = np.array([self._safemean(scaled[:, d, :]) for d in range(y_true.shape[1])])
        return float(np.mean(per_dim))


class RMSE(Metric):
    """Root mean squared error."""

    def compute(
        self,
        *,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_past: np.ndarray,
        y_past_indptr: np.ndarray,
        q_pred: np.ndarray,
        seasonality: int,
        quantile_levels: list[float],
    ) -> float:
        per_dim = np.sqrt(np.nanmean((y_true - y_pred) ** 2, axis=(0, 2)))  # [D]
        return float(np.mean(per_dim))


class RMSSE(Metric):
    """Root mean squared scaled error.

    Warning:
        Items with undefined in-sample seasonal error (e.g., history shorter than `seasonality`,
        all-NaN history, or zero seasonal error) are excluded from aggregation.
    """

    def __init__(self, epsilon: float = 0.0) -> None:
        self.epsilon = epsilon

    def compute(
        self,
        *,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_past: np.ndarray,
        y_past_indptr: np.ndarray,
        q_pred: np.ndarray,
        seasonality: int,
        quantile_levels: list[float],
    ) -> float:
        seasonal_error = _squared_seasonal_error(y_past=y_past, indptr=y_past_indptr, seasonality=seasonality)  # [N, D]
        seasonal_error = np.clip(seasonal_error, self.epsilon, None)
        scaled = (y_true - y_pred) ** 2 / seasonal_error[:, :, None]  # [N, D, H]
        per_dim = np.array([np.sqrt(self._safemean(scaled[:, d, :])) for d in range(y_true.shape[1])])
        return float(np.mean(per_dim))


class MSE(Metric):
    """Mean squared error."""

    def compute(
        self,
        *,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_past: np.ndarray,
        y_past_indptr: np.ndarray,
        q_pred: np.ndarray,
        seasonality: int,
        quantile_levels: list[float],
    ) -> float:
        per_dim = np.nanmean((y_true - y_pred) ** 2, axis=(0, 2))  # [D]
        return float(np.mean(per_dim))


class RMSLE(Metric):
    """Root mean squared logarithmic error."""

    def compute(
        self,
        *,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_past: np.ndarray,
        y_past_indptr: np.ndarray,
        q_pred: np.ndarray,
        seasonality: int,
        quantile_levels: list[float],
    ) -> float:
        per_dim = np.sqrt(np.nanmean((np.log1p(y_true) - np.log1p(y_pred)) ** 2, axis=(0, 2)))  # [D]
        return float(np.mean(per_dim))


class MAPE(Metric):
    """Mean absolute percentage error."""

    def compute(
        self,
        *,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_past: np.ndarray,
        y_past_indptr: np.ndarray,
        q_pred: np.ndarray,
        seasonality: int,
        quantile_levels: list[float],
    ) -> float:
        ratio = np.abs(y_true - y_pred) / np.abs(y_true)  # [N, D, H]
        per_dim = np.array([self._safemean(ratio[:, d, :]) for d in range(y_true.shape[1])])
        return float(np.mean(per_dim))


class SMAPE(Metric):
    """Symmetric mean absolute percentage error."""

    def compute(
        self,
        *,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_past: np.ndarray,
        y_past_indptr: np.ndarray,
        q_pred: np.ndarray,
        seasonality: int,
        quantile_levels: list[float],
    ) -> float:
        val = 2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))  # [N, D, H]
        per_dim = np.array([self._safemean(val[:, d, :]) for d in range(y_true.shape[1])])
        return float(np.mean(per_dim))


class MQL(Metric):
    """Mean quantile loss."""

    needs_quantiles: bool = True

    def compute(
        self,
        *,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_past: np.ndarray,
        y_past_indptr: np.ndarray,
        q_pred: np.ndarray,
        seasonality: int,
        quantile_levels: list[float],
    ) -> float:
        if len(quantile_levels) == 0:
            raise ValueError(f"{self.__class__.__name__} cannot be computed without quantile_levels")
        ql = _quantile_loss(y_true=y_true, q_pred=q_pred, quantile_levels=quantile_levels)  # [N, D, H, Q]
        per_dim = np.nanmean(ql, axis=(0, 2, 3))  # [D]
        return float(np.mean(per_dim))


class SQL(Metric):
    """Scaled quantile loss.

    Warning:
        Items with undefined in-sample seasonal error (e.g., history shorter than `seasonality`,
        all-NaN history, or zero seasonal error) are excluded from aggregation.
    """

    needs_quantiles: bool = True

    def __init__(self, epsilon: float = 0.0) -> None:
        self.epsilon = epsilon

    def compute(
        self,
        *,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_past: np.ndarray,
        y_past_indptr: np.ndarray,
        q_pred: np.ndarray,
        seasonality: int,
        quantile_levels: list[float],
    ) -> float:
        ql = _quantile_loss(y_true=y_true, q_pred=q_pred, quantile_levels=quantile_levels)  # [N, D, H, Q]
        ql_avg_q = np.nanmean(ql, axis=3)  # [N, D, H] — average over quantiles
        seasonal_error = _abs_seasonal_error(y_past=y_past, indptr=y_past_indptr, seasonality=seasonality)  # [N, D]
        seasonal_error = np.clip(seasonal_error, self.epsilon, None)
        scaled = ql_avg_q / seasonal_error[:, :, None]  # [N, D, H]
        per_dim = np.array([self._safemean(scaled[:, d, :]) for d in range(y_true.shape[1])])
        return float(np.mean(per_dim))


class WQL(Metric):
    """Weighted quantile loss."""

    needs_quantiles: bool = True

    def __init__(self, epsilon: float = 0.0) -> None:
        self.epsilon = epsilon

    def compute(
        self,
        *,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_past: np.ndarray,
        y_past_indptr: np.ndarray,
        q_pred: np.ndarray,
        seasonality: int,
        quantile_levels: list[float],
    ) -> float:
        ql = _quantile_loss(y_true=y_true, q_pred=q_pred, quantile_levels=quantile_levels)  # [N, D, H, Q]
        ql_per_dim = np.nanmean(ql, axis=(0, 2, 3))  # [D]
        abs_true_per_dim = np.nanmean(np.abs(y_true), axis=(0, 2))  # [D]
        per_dim = ql_per_dim / np.maximum(abs_true_per_dim, self.epsilon)
        return float(np.mean(per_dim))


def _quantile_loss(
    *,
    y_true: np.ndarray,
    q_pred: np.ndarray,
    quantile_levels: list[float],
) -> np.ndarray:
    """Compute quantile loss.

    Returns
    -------
    np.ndarray [N, D, H, Q]
    """
    y_true_expanded = y_true[..., None]  # [N, D, H, 1]
    q_arr = np.array(quantile_levels)  # [Q]
    return 2 * np.abs((y_true_expanded - q_pred) * ((y_true_expanded <= q_pred) - q_arr))


def _seasonal_error(
    *,
    y_past: np.ndarray,
    indptr: np.ndarray,
    seasonality: int,
    aggregate_fn: Callable,
) -> np.ndarray:
    """Compute seasonal error for each (item, dim) pair.

    Parameters
    ----------
    y_past : np.ndarray [total_T, D]
        Concatenated past observations.
    indptr : np.ndarray [N+1]
        CSR-style index pointer. Item i has y_past[indptr[i]:indptr[i+1], :].
    seasonality : int
        Seasonal period.
    aggregate_fn : Callable
        Applied element-wise to seasonal diffs (e.g. np.abs or np.square).

    Returns
    -------
    np.ndarray [N, D]
    """
    num_series = len(indptr) - 1
    num_dims = y_past.shape[1]

    if num_series == 0:
        return np.array([], dtype="float64").reshape(0, 0)

    lengths = np.diff(indptr)
    num_diffs_per_series = np.maximum(lengths - seasonality, 0)

    if num_diffs_per_series.sum() == 0:
        return np.full((num_series, num_dims), np.nan, dtype="float64")

    total_diffs = int(num_diffs_per_series.sum())
    series_ids = np.repeat(np.arange(num_series, dtype=np.int64), num_diffs_per_series)
    diff_offsets = np.arange(total_diffs) - np.repeat(
        np.cumsum(num_diffs_per_series) - num_diffs_per_series, num_diffs_per_series
    )

    idx_current = indptr[series_ids] + seasonality + diff_offsets
    idx_lagged = idx_current - seasonality

    diffs = y_past[idx_current] - y_past[idx_lagged]  # [total_diffs, D]
    errors = aggregate_fn(diffs)  # [total_diffs, D]

    valid = ~np.isnan(errors)  # [total_diffs, D]
    result = np.full((num_series, num_dims), np.nan, dtype="float64")
    for d in range(num_dims):
        sums = np.bincount(series_ids, weights=np.where(valid[:, d], errors[:, d], 0.0), minlength=num_series)
        counts = np.bincount(series_ids, weights=valid[:, d].astype("float64"), minlength=num_series)
        mask = counts > 0
        result[mask, d] = sums[mask] / counts[mask]

    return result


def _abs_seasonal_error(*, y_past: np.ndarray, indptr: np.ndarray, seasonality: int) -> np.ndarray:
    """Compute mean absolute seasonal error. Returns [N, D]."""
    return _seasonal_error(y_past=y_past, indptr=indptr, seasonality=seasonality, aggregate_fn=np.abs)


def _squared_seasonal_error(*, y_past: np.ndarray, indptr: np.ndarray, seasonality: int) -> np.ndarray:
    """Compute mean squared seasonal error. Returns [N, D]."""
    return _seasonal_error(y_past=y_past, indptr=indptr, seasonality=seasonality, aggregate_fn=np.square)


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

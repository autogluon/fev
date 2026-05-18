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
    def _safemean(arr: np.ndarray, axis=None) -> float | np.ndarray:
        """Compute mean ignoring NaN, Inf, and -Inf values."""
        mask = ~np.isfinite(arr)
        if mask.any():
            arr = np.where(mask, np.nan, arr)
        return np.nanmean(arr, axis=axis)

    def compute(
        self,
        *,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_past: np.ndarray,
        y_past_lengths: np.ndarray,
        q_pred: np.ndarray,
        seasonality: int,
        quantile_levels: list[float],
    ) -> float:
        """Compute the metric score. Computed per target dim, then averaged across dims.

        Parameters
        ----------
        y_true : np.ndarray [N, H, D]
            Ground truth. N=number of time series, H=forecast horizon, D=target dimensions.
        y_pred : np.ndarray [N, H, D]
            Point forecast predictions, same shape as y_true.
        y_past : np.ndarray [total_T, D]
            Concatenated historical observations for all items (ragged time axis).
        y_past_lengths : np.ndarray [N]
            Number of past observations per item. sum(y_past_lengths) == total_T.
        q_pred : np.ndarray [N, H, D, Q]
            Quantile predictions. Q=len(quantile_levels), or Q=0 if none requested.
        seasonality : int
            Seasonal period for scaled error metrics (MASE, RMSSE, SQL).
        quantile_levels : list[float]
            Quantile levels in (0, 1) corresponding to q_pred's last axis.
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
        y_past_lengths: np.ndarray,
        q_pred: np.ndarray,
        seasonality: int,
        quantile_levels: list[float],
    ) -> float:
        per_dim = np.nanmean(np.abs(y_true - y_pred), axis=(0, 1))  # [D]
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
        y_past_lengths: np.ndarray,
        q_pred: np.ndarray,
        seasonality: int,
        quantile_levels: list[float],
    ) -> float:
        abs_err_per_dim = np.nanmean(np.abs(y_true - y_pred), axis=(0, 1))  # [D]
        abs_true_per_dim = np.nanmean(np.abs(y_true), axis=(0, 1))  # [D]
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
        y_past_lengths: np.ndarray,
        q_pred: np.ndarray,
        seasonality: int,
        quantile_levels: list[float],
    ) -> float:
        seasonal_error = _abs_seasonal_error_per_item(
            y_past=y_past, y_past_lengths=y_past_lengths, seasonality=seasonality
        )  # [N, D]
        seasonal_error = np.clip(seasonal_error, self.epsilon, None)
        scaled = np.abs(y_true - y_pred) / seasonal_error[:, None, :]  # [N, H, D]
        return float(np.mean(self._safemean(scaled, axis=(0, 1))))


class RMSE(Metric):
    """Root mean squared error."""

    def compute(
        self,
        *,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_past: np.ndarray,
        y_past_lengths: np.ndarray,
        q_pred: np.ndarray,
        seasonality: int,
        quantile_levels: list[float],
    ) -> float:
        per_dim = np.sqrt(np.nanmean((y_true - y_pred) ** 2, axis=(0, 1)))  # [D]
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
        y_past_lengths: np.ndarray,
        q_pred: np.ndarray,
        seasonality: int,
        quantile_levels: list[float],
    ) -> float:
        seasonal_error = _squared_seasonal_error_per_item(
            y_past=y_past, y_past_lengths=y_past_lengths, seasonality=seasonality
        )  # [N, D]
        seasonal_error = np.clip(seasonal_error, self.epsilon, None)
        scaled = (y_true - y_pred) ** 2 / seasonal_error[:, None, :]  # [N, H, D]
        return float(np.mean(np.sqrt(self._safemean(scaled, axis=(0, 1)))))


class MSE(Metric):
    """Mean squared error."""

    def compute(
        self,
        *,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_past: np.ndarray,
        y_past_lengths: np.ndarray,
        q_pred: np.ndarray,
        seasonality: int,
        quantile_levels: list[float],
    ) -> float:
        per_dim = np.nanmean((y_true - y_pred) ** 2, axis=(0, 1))  # [D]
        return float(np.mean(per_dim))


class RMSLE(Metric):
    """Root mean squared logarithmic error."""

    def compute(
        self,
        *,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_past: np.ndarray,
        y_past_lengths: np.ndarray,
        q_pred: np.ndarray,
        seasonality: int,
        quantile_levels: list[float],
    ) -> float:
        per_dim = np.sqrt(np.nanmean((np.log1p(y_true) - np.log1p(y_pred)) ** 2, axis=(0, 1)))  # [D]
        return float(np.mean(per_dim))


class MAPE(Metric):
    """Mean absolute percentage error."""

    def compute(
        self,
        *,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_past: np.ndarray,
        y_past_lengths: np.ndarray,
        q_pred: np.ndarray,
        seasonality: int,
        quantile_levels: list[float],
    ) -> float:
        ratio = np.abs(y_true - y_pred) / np.abs(y_true)  # [N, H, D]
        return float(np.mean(self._safemean(ratio, axis=(0, 1))))


class SMAPE(Metric):
    """Symmetric mean absolute percentage error."""

    def compute(
        self,
        *,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_past: np.ndarray,
        y_past_lengths: np.ndarray,
        q_pred: np.ndarray,
        seasonality: int,
        quantile_levels: list[float],
    ) -> float:
        val = 2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))  # [N, H, D]
        return float(np.mean(self._safemean(val, axis=(0, 1))))


class MQL(Metric):
    """Mean quantile loss."""

    needs_quantiles: bool = True

    def compute(
        self,
        *,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_past: np.ndarray,
        y_past_lengths: np.ndarray,
        q_pred: np.ndarray,
        seasonality: int,
        quantile_levels: list[float],
    ) -> float:
        if len(quantile_levels) == 0:
            raise ValueError(f"{self.__class__.__name__} cannot be computed without quantile_levels")
        ql = _quantile_loss(y_true=y_true, q_pred=q_pred, quantile_levels=quantile_levels)  # [N, H, D, Q]
        per_dim = np.nanmean(ql, axis=(0, 1, 3))  # [D]
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
        y_past_lengths: np.ndarray,
        q_pred: np.ndarray,
        seasonality: int,
        quantile_levels: list[float],
    ) -> float:
        ql = _quantile_loss(y_true=y_true, q_pred=q_pred, quantile_levels=quantile_levels)  # [N, H, D, Q]
        ql_avg_q = np.nanmean(ql, axis=3)  # [N, H, D]
        seasonal_error = _abs_seasonal_error_per_item(
            y_past=y_past, y_past_lengths=y_past_lengths, seasonality=seasonality
        )  # [N, D]
        seasonal_error = np.clip(seasonal_error, self.epsilon, None)
        scaled = ql_avg_q / seasonal_error[:, None, :]  # [N, H, D]
        return float(np.mean(self._safemean(scaled, axis=(0, 1))))


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
        y_past_lengths: np.ndarray,
        q_pred: np.ndarray,
        seasonality: int,
        quantile_levels: list[float],
    ) -> float:
        ql = _quantile_loss(y_true=y_true, q_pred=q_pred, quantile_levels=quantile_levels)  # [N, H, D, Q]
        ql_per_dim = np.nanmean(ql, axis=(0, 1, 3))  # [D]
        abs_true_per_dim = np.nanmean(np.abs(y_true), axis=(0, 1))  # [D]
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
    np.ndarray [N, H, D, Q]
    """
    y_true_expanded = y_true[..., None]  # [N, H, D, 1]
    q_arr = np.array(quantile_levels)  # [Q]
    return 2 * np.abs((y_true_expanded - q_pred) * ((y_true_expanded <= q_pred) - q_arr))


def _seasonal_error_per_item(
    *,
    y_past: np.ndarray,
    y_past_lengths: np.ndarray,
    seasonality: int,
    aggregate_fn: Callable,
) -> np.ndarray:
    """Compute seasonal error for each (item, dim) pair.

    Parameters
    ----------
    y_past : np.ndarray [total_T, D]
        Concatenated past observations.
    y_past_lengths : np.ndarray [N]
        Number of observations per item.
    seasonality : int
        Seasonal period.
    aggregate_fn : Callable
        Applied element-wise to seasonal diffs (e.g. np.abs or np.square).

    Returns
    -------
    np.ndarray [N, D]
    """
    num_series = len(y_past_lengths)
    num_dims = y_past.shape[1]

    if num_series == 0:
        return np.array([], dtype="float64").reshape(0, 0)

    num_diffs_per_series = np.maximum(y_past_lengths - seasonality, 0)

    if num_diffs_per_series.sum() == 0:
        return np.full((num_series, num_dims), np.nan, dtype="float64")

    # Fast path: all items have equal length — reshape + slice instead of fancy indexing
    if np.all(y_past_lengths == y_past_lengths[0]):
        T = int(y_past_lengths[0])
        y_reshaped = y_past.reshape(num_series, T, num_dims)
        diffs = y_reshaped[:, seasonality:, :] - y_reshaped[:, :-seasonality, :]
        return np.nanmean(aggregate_fn(diffs), axis=1)

    total_diffs = int(num_diffs_per_series.sum())
    series_ids = np.repeat(np.arange(num_series, dtype=np.int64), num_diffs_per_series)
    diff_offsets = np.arange(total_diffs) - np.repeat(
        np.cumsum(num_diffs_per_series) - num_diffs_per_series, num_diffs_per_series
    )

    offsets = np.empty(num_series + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(y_past_lengths, out=offsets[1:])
    idx_current = offsets[series_ids] + seasonality + diff_offsets
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


def _abs_seasonal_error_per_item(*, y_past: np.ndarray, y_past_lengths: np.ndarray, seasonality: int) -> np.ndarray:
    """Compute mean absolute seasonal error. Returns [N, D]."""
    return _seasonal_error_per_item(
        y_past=y_past, y_past_lengths=y_past_lengths, seasonality=seasonality, aggregate_fn=np.abs
    )


def _squared_seasonal_error_per_item(
    *, y_past: np.ndarray, y_past_lengths: np.ndarray, seasonality: int
) -> np.ndarray:
    """Compute mean squared seasonal error. Returns [N, D]."""
    return _seasonal_error_per_item(
        y_past=y_past, y_past_lengths=y_past_lengths, seasonality=seasonality, aggregate_fn=np.square
    )


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

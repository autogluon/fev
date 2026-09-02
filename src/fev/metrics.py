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
        reduce_axes: tuple[int, ...] = (0, 1),
    ) -> np.ndarray:
        """Compute the metric per target dim, reducing over `reduce_axes`.

        The target-dim axis is always kept; callers average over it (or keep it for per-target scores).

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
        reduce_axes : tuple[int, ...], default (0, 1)
            Which of the item (axis 0) and horizon (axis 1) axes to aggregate over. `(0, 1)` yields an
            overall per-dim score `[D]`; `(1,)` yields a per-item per-dim score `[N, D]`.

        Returns
        -------
        np.ndarray
            Per-dim scores. Shape `[D]` for `reduce_axes=(0, 1)`, `[N, D]` for `reduce_axes=(1,)`.
        """
        raise NotImplementedError

    def compute_scores(
        self,
        *,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_past: np.ndarray,
        y_past_lengths: np.ndarray,
        q_pred: np.ndarray,
        seasonality: int,
        quantile_levels: list[float],
        reduce_axes: tuple[int, ...] = (0, 1),
        per_quantile_scores: bool = False,
    ) -> dict[str, np.ndarray]:
        """Named per-dim scores for this metric, reduced over `reduce_axes`. Returns `{self.name: self.compute(...)}`."""
        return {
            self.name: self.compute(
                y_true=y_true,
                y_pred=y_pred,
                y_past=y_past,
                y_past_lengths=y_past_lengths,
                q_pred=q_pred,
                seasonality=seasonality,
                quantile_levels=quantile_levels,
                reduce_axes=reduce_axes,
            )
        }


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
        reduce_axes: tuple[int, ...] = (0, 1),
    ) -> np.ndarray:
        return np.nanmean(np.abs(y_true - y_pred), axis=reduce_axes)


class MAEB(Metric):
    """Mean absolute error plus an absolute mean bias penalty. Equals MAE when the forecast is unbiased."""

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
        reduce_axes: tuple[int, ...] = (0, 1),
    ) -> np.ndarray:
        abs_err = np.nanmean(np.abs(y_true - y_pred), axis=reduce_axes)
        bias = np.nanmean(y_pred - y_true, axis=reduce_axes)
        return abs_err + np.abs(bias)


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
        reduce_axes: tuple[int, ...] = (0, 1),
    ) -> np.ndarray:
        abs_err = np.nanmean(np.abs(y_true - y_pred), axis=reduce_axes)
        abs_true = np.nanmean(np.abs(y_true), axis=reduce_axes)
        return abs_err / np.maximum(abs_true, self.epsilon)


class WAPEB(Metric):
    """Weighted absolute percentage error plus an absolute bias penalty (scale-free MAEB; VN1 challenge metric).

    Equals WAPE when the forecast is unbiased.
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
        reduce_axes: tuple[int, ...] = (0, 1),
    ) -> np.ndarray:
        abs_err = np.nanmean(np.abs(y_true - y_pred), axis=reduce_axes)
        bias = np.nanmean(y_pred - y_true, axis=reduce_axes)
        abs_true = np.nanmean(np.abs(y_true), axis=reduce_axes)
        return (abs_err + np.abs(bias)) / np.maximum(abs_true, self.epsilon)


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
        reduce_axes: tuple[int, ...] = (0, 1),
    ) -> np.ndarray:
        seasonal_error = _abs_seasonal_error_per_item(
            y_past=y_past, y_past_lengths=y_past_lengths, seasonality=seasonality
        )  # [N, D]
        seasonal_error = np.clip(seasonal_error, self.epsilon, None)
        scaled = np.abs(y_true - y_pred) / seasonal_error[:, None, :]  # [N, H, D]
        return self._safemean(scaled, axis=reduce_axes)


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
        reduce_axes: tuple[int, ...] = (0, 1),
    ) -> np.ndarray:
        return np.sqrt(np.nanmean((y_true - y_pred) ** 2, axis=reduce_axes))


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
        reduce_axes: tuple[int, ...] = (0, 1),
    ) -> np.ndarray:
        seasonal_error = _squared_seasonal_error_per_item(
            y_past=y_past, y_past_lengths=y_past_lengths, seasonality=seasonality
        )  # [N, D]
        seasonal_error = np.clip(seasonal_error, self.epsilon, None)
        scaled = (y_true - y_pred) ** 2 / seasonal_error[:, None, :]  # [N, H, D]
        return np.sqrt(self._safemean(scaled, axis=reduce_axes))


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
        reduce_axes: tuple[int, ...] = (0, 1),
    ) -> np.ndarray:
        return np.nanmean((y_true - y_pred) ** 2, axis=reduce_axes)


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
        reduce_axes: tuple[int, ...] = (0, 1),
    ) -> np.ndarray:
        return np.sqrt(np.nanmean((np.log1p(y_true) - np.log1p(y_pred)) ** 2, axis=reduce_axes))


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
        reduce_axes: tuple[int, ...] = (0, 1),
    ) -> np.ndarray:
        ratio = np.abs(y_true - y_pred) / np.abs(y_true)  # [N, H, D]
        return self._safemean(ratio, axis=reduce_axes)


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
        reduce_axes: tuple[int, ...] = (0, 1),
    ) -> np.ndarray:
        val = 2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))  # [N, H, D]
        return self._safemean(val, axis=reduce_axes)


class QuantileMetric(Metric):
    """Base class for quantile loss metrics (MQL, WQL, SQL, NZQL).

    Subclasses implement `_per_quantile_level`. The overall score is the mean over quantile levels,
    so `SQL` always equals the mean of `SQL[0.1], SQL[0.5], ...` (single code path, cannot drift).
    """

    needs_quantiles: bool = True

    def _per_quantile_level(
        self,
        *,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_past: np.ndarray,
        y_past_lengths: np.ndarray,
        q_pred: np.ndarray,
        seasonality: int,
        quantile_levels: list[float],
        reduce_axes: tuple[int, ...],
    ) -> np.ndarray:
        """Compute the metric per dim and quantile level, reduced over `reduce_axes`.

        Returns `[D, Q]` for `reduce_axes=(0, 1)`, `[N, D, Q]` for `reduce_axes=(1,)`.
        """
        raise NotImplementedError

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
        reduce_axes: tuple[int, ...] = (0, 1),
    ) -> np.ndarray:
        if len(quantile_levels) == 0:
            raise ValueError(f"{self.name} cannot be computed without quantile_levels")
        per_level = self._per_quantile_level(
            y_true=y_true,
            y_pred=y_pred,
            y_past=y_past,
            y_past_lengths=y_past_lengths,
            q_pred=q_pred,
            seasonality=seasonality,
            quantile_levels=quantile_levels,
            reduce_axes=reduce_axes,
        )  # [..., D, Q]
        return np.mean(per_level, axis=-1)  # mean over quantile levels -> [..., D]

    def compute_scores(
        self,
        *,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_past: np.ndarray,
        y_past_lengths: np.ndarray,
        q_pred: np.ndarray,
        seasonality: int,
        quantile_levels: list[float],
        reduce_axes: tuple[int, ...] = (0, 1),
        per_quantile_scores: bool = False,
    ) -> dict[str, np.ndarray]:
        if len(quantile_levels) == 0:
            raise ValueError(f"{self.name} cannot be computed without quantile_levels")
        per_level = self._per_quantile_level(
            y_true=y_true,
            y_pred=y_pred,
            y_past=y_past,
            y_past_lengths=y_past_lengths,
            q_pred=q_pred,
            seasonality=seasonality,
            quantile_levels=quantile_levels,
            reduce_axes=reduce_axes,
        )  # [..., D, Q]
        assert per_level.shape[-1] == len(quantile_levels)
        scores = {self.name: np.mean(per_level, axis=-1)}
        if per_quantile_scores:
            scores.update({f"{self.name}[{q}]": per_level[..., i] for i, q in enumerate(quantile_levels)})
        return scores


class MQL(QuantileMetric):
    """Mean quantile loss."""

    def _per_quantile_level(
        self,
        *,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_past: np.ndarray,
        y_past_lengths: np.ndarray,
        q_pred: np.ndarray,
        seasonality: int,
        quantile_levels: list[float],
        reduce_axes: tuple[int, ...],
    ) -> np.ndarray:
        ql = _quantile_loss(y_true=y_true, q_pred=q_pred, quantile_levels=quantile_levels)  # [N, H, D, Q]
        return np.nanmean(ql, axis=reduce_axes)  # [..., D, Q]


class SQL(QuantileMetric):
    """Scaled quantile loss.

    Warning:
        Items with undefined in-sample seasonal error (e.g., history shorter than `seasonality`,
        all-NaN history, or zero seasonal error) are excluded from aggregation.
    """

    def __init__(self, epsilon: float = 0.0) -> None:
        self.epsilon = epsilon

    def _per_quantile_level(
        self,
        *,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_past: np.ndarray,
        y_past_lengths: np.ndarray,
        q_pred: np.ndarray,
        seasonality: int,
        quantile_levels: list[float],
        reduce_axes: tuple[int, ...],
    ) -> np.ndarray:
        ql = _quantile_loss(y_true=y_true, q_pred=q_pred, quantile_levels=quantile_levels)  # [N, H, D, Q]
        seasonal_error = _abs_seasonal_error_per_item(
            y_past=y_past, y_past_lengths=y_past_lengths, seasonality=seasonality
        )  # [N, D]
        seasonal_error = np.clip(seasonal_error, self.epsilon, None)
        scaled = ql / seasonal_error[:, None, :, None]  # [N, H, D, Q]
        return self._safemean(scaled, axis=reduce_axes)  # [..., D, Q]


class NZQL(QuantileMetric):
    """Quantile loss normalized by each item's average non-zero historical magnitude.

    Each quantile loss is divided by the mean absolute value of non-zero historical observations
    for its item and target dimension. Items with no non-zero historical observations are excluded
    from aggregation.
    """

    def __init__(self, epsilon: float = 0.0) -> None:
        self.epsilon = epsilon

    @staticmethod
    def _mean_nonzero_abs_history_per_item(y_past: np.ndarray, y_past_lengths: np.ndarray) -> np.ndarray:
        """Per-item mean absolute non-zero history value. Returns [N, D]; NaN where none exist."""
        num_series = len(y_past_lengths)
        num_dims = y_past.shape[1]
        result = np.full((num_series, num_dims), np.nan, dtype="float64")

        starts = np.concatenate(([0], np.cumsum(y_past_lengths)[:-1]))
        valid_series = y_past_lengths > 0
        if not valid_series.any():
            return result

        nonzero = ~np.isnan(y_past) & (y_past != 0)
        sums = np.add.reduceat(np.where(nonzero, np.abs(y_past), 0.0), starts[valid_series], axis=0)
        counts = np.add.reduceat(nonzero.astype(np.int64), starts[valid_series], axis=0)
        result[valid_series] = np.divide(sums, counts, where=counts > 0, out=np.full_like(sums, np.nan))
        return result

    def _per_quantile_level(
        self,
        *,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_past: np.ndarray,
        y_past_lengths: np.ndarray,
        q_pred: np.ndarray,
        seasonality: int,
        quantile_levels: list[float],
        reduce_axes: tuple[int, ...],
    ) -> np.ndarray:
        ql = _quantile_loss(y_true=y_true, q_pred=q_pred, quantile_levels=quantile_levels)  # [N, H, D, Q]
        scale = self._mean_nonzero_abs_history_per_item(y_past, y_past_lengths)  # [N, D]
        scale = np.clip(scale, self.epsilon, None)
        scaled = ql / scale[:, None, :, None]  # [N, H, D, Q]
        return self._safemean(scaled, axis=reduce_axes)  # [..., D, Q]


class WQL(QuantileMetric):
    """Weighted quantile loss."""

    def __init__(self, epsilon: float = 0.0) -> None:
        self.epsilon = epsilon

    def _per_quantile_level(
        self,
        *,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_past: np.ndarray,
        y_past_lengths: np.ndarray,
        q_pred: np.ndarray,
        seasonality: int,
        quantile_levels: list[float],
        reduce_axes: tuple[int, ...],
    ) -> np.ndarray:
        ql = _quantile_loss(y_true=y_true, q_pred=q_pred, quantile_levels=quantile_levels)  # [N, H, D, Q]
        ql_agg = np.nanmean(ql, axis=reduce_axes)  # [..., D, Q]
        abs_true = np.nanmean(np.abs(y_true), axis=reduce_axes)  # [..., D]
        return ql_agg / np.maximum(abs_true, self.epsilon)[..., None]  # [..., D, Q]


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
    # Bias-penalized errors
    "MAEB": MAEB,
    "WAPEB": WAPEB,
    # Logarithmic errors
    "RMSLE": RMSLE,
    # Percentage errors
    "MAPE": MAPE,
    "SMAPE": SMAPE,
    # Quantile loss
    "MQL": MQL,
    "WQL": WQL,
    "SQL": SQL,
    "NZQL": NZQL,
}

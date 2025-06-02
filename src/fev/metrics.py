from typing import Type

import datasets
import numpy as np

from fev.constants import DEFAULT_NUM_PROC, PREDICTIONS


class Metric:
    """Base class for all metrics."""

    @staticmethod
    def _safemean(arr: np.ndarray) -> float:
        """Compute mean of an array, ignoring NaN, Inf, and -Inf values."""
        return np.mean(arr[np.isfinite(arr)])

    def compute(
        self,
        *,
        test_data: datasets.Dataset,
        predictions: datasets.Dataset,
        past_data: datasets.Dataset,
        seasonality: int = 1,
        quantile_levels: list[float] | None = None,
        target_column: str = "target",
    ) -> float:
        raise NotImplementedError


class MAE(Metric):
    """Mean absolute error."""

    def compute(
        self,
        *,
        test_data: datasets.Dataset,
        predictions: datasets.Dataset,
        past_data: datasets.Dataset,
        seasonality: int = 1,
        quantile_levels: list[float] | None = None,
        target_column: str = "target",
    ):
        y_test = np.array(test_data[target_column])
        y_pred = np.array(predictions[PREDICTIONS])
        return np.nanmean(np.abs(y_test - y_pred))


class WAPE(Metric):
    """Weighted absolute percentage error."""

    def compute(
        self,
        *,
        test_data: datasets.Dataset,
        predictions: datasets.Dataset,
        past_data: datasets.Dataset,
        seasonality: int = 1,
        quantile_levels: list[float] | None = None,
        target_column: str = "target",
    ):
        y_test = np.array(test_data[target_column])
        y_pred = np.array(predictions[PREDICTIONS])
        return np.nanmean(np.abs(y_test - y_pred)) / np.nanmean(np.abs(y_test))


class MASE(Metric):
    """Mean absolute scaled error."""

    def compute(
        self,
        *,
        test_data: datasets.Dataset,
        predictions: datasets.Dataset,
        past_data: datasets.Dataset,
        seasonality: int = 1,
        quantile_levels: list[float] | None = None,
        target_column: str = "target",
    ):
        y_test = np.array(test_data[target_column])
        y_pred = np.array(predictions[PREDICTIONS])

        seasonal_error = _abs_seasonal_error_per_item(
            past_data=past_data, seasonality=seasonality, target_column=target_column
        )
        return self._safemean(np.abs(y_test - y_pred) / seasonal_error[:, None])


class RMSE(Metric):
    """Root mean squared error."""

    def compute(
        self,
        *,
        test_data: datasets.Dataset,
        predictions: datasets.Dataset,
        past_data: datasets.Dataset,
        seasonality: int = 1,
        quantile_levels: list[float] | None = None,
        target_column: str = "target",
    ):
        y_test = np.array(test_data[target_column])
        y_pred = np.array(predictions[PREDICTIONS])
        return np.sqrt(np.nanmean((y_test - y_pred) ** 2))


class RMSSE(Metric):
    """Root mean squared scaled error."""

    def compute(
        self,
        *,
        test_data: datasets.Dataset,
        predictions: datasets.Dataset,
        past_data: datasets.Dataset,
        seasonality: int = 1,
        quantile_levels: list[float] | None = None,
        target_column: str = "target",
    ):
        y_test = np.array(test_data[target_column])
        y_pred = np.array(predictions[PREDICTIONS])
        seasonal_error = _squared_seasonal_error_per_item(
            past_data, seasonality=seasonality, target_column=target_column
        )
        return np.sqrt(self._safemean((y_test - y_pred) ** 2 / seasonal_error[:, None]))


class MSE(Metric):
    """Mean squared error."""

    def compute(
        self,
        *,
        test_data: datasets.Dataset,
        predictions: datasets.Dataset,
        past_data: datasets.Dataset,
        seasonality: int = 1,
        quantile_levels: list[float] | None = None,
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
        seasonality: int = 1,
        quantile_levels: list[float] | None = None,
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
        seasonality: int = 1,
        quantile_levels: list[float] | None = None,
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
        seasonality: int = 1,
        quantile_levels: list[float] | None = None,
        target_column: str = "target",
    ):
        y_test = np.array(test_data[target_column])
        y_pred = np.array(predictions[PREDICTIONS])
        return self._safemean(2 * np.abs(y_test - y_pred) / (np.abs(y_test) + np.abs(y_pred)))


class MQL(Metric):
    """Mean quantile loss."""

    def compute(
        self,
        *,
        test_data: datasets.Dataset,
        predictions: datasets.Dataset,
        past_data: datasets.Dataset,
        seasonality: int = 1,
        quantile_levels: list[float] | None = None,
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

    def compute(
        self,
        *,
        test_data: datasets.Dataset,
        predictions: datasets.Dataset,
        past_data: datasets.Dataset,
        seasonality: int = 1,
        quantile_levels: list[float] | None = None,
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
        return self._safemean(ql_per_time_step / seasonal_error[:, None])


class WQL(Metric):
    """Weighted quantile loss."""

    def compute(
        self,
        *,
        test_data: datasets.Dataset,
        predictions: datasets.Dataset,
        past_data: datasets.Dataset,
        seasonality: int = 1,
        quantile_levels: list[float] | None = None,
        target_column: str = "target",
    ):
        ql = _quantile_loss(
            test_data=test_data,
            predictions=predictions,
            quantile_levels=quantile_levels,
            target_column=target_column,
        )
        return np.nanmean(ql) / np.nanmean(np.abs(np.array(test_data[target_column])))


def _seasonal_diff(array: np.ndarray, seasonality: int) -> np.ndarray:
    if len(array) <= seasonality:
        return np.array([])
    else:
        return array[seasonality:] - array[:-seasonality]


def _abs_seasonal_error(array: np.ndarray, seasonality: int) -> float:
    return np.nanmean(np.abs(_seasonal_diff(array=array, seasonality=seasonality)))


def _squared_seasonal_error(array: np.ndarray, seasonality: int) -> float:
    return np.nanmean(_seasonal_diff(array=array, seasonality=seasonality) ** 2)


def _abs_seasonal_error_per_item(
    past_data: datasets.Dataset, seasonality: int, target_column: str, nan_fill_value: float = 1.0
) -> np.ndarray:
    """Compute mean absolute seasonal error for each time series in past_data."""
    abs_seasonal_error = past_data.map(
        lambda record: {"_abs_seasonal_error": float(_abs_seasonal_error(record[target_column], seasonality))},
        num_proc=min(DEFAULT_NUM_PROC, len(past_data)),
    )["_abs_seasonal_error"]
    return np.nan_to_num(abs_seasonal_error, nan=nan_fill_value).astype("float64")


def _squared_seasonal_error_per_item(
    past_data: datasets.Dataset, seasonality: int, target_column: str, nan_fill_value: float = 1.0
) -> np.ndarray:
    """Compute mean squared seasonal error for each time series in past_data."""
    squared_seasonal_error = past_data.map(
        lambda record: {"_squared_seasonal_error": float(_squared_seasonal_error(record[target_column], seasonality))},
        num_proc=min(DEFAULT_NUM_PROC, len(past_data)),
    )["_squared_seasonal_error"]
    return np.nan_to_num(squared_seasonal_error, nan=nan_fill_value).astype("float64")


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
    # # Quantile loss
    "MQL": MQL,
    "WQL": WQL,
    "SQL": SQL,
}

QUANTILE_METRICS = ["MQL", "WQL", "SQL"]

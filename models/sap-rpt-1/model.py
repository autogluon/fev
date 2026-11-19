"""SAP RPT-1 model wrapper for fev-bench (v0.8.0 format).

SAP RPT-1 is a tabular foundation model applied to time series forecasting
via per-item tabular regression with temporal feature engineering.
"""

import warnings

import datasets
import numpy as np
import pandas as pd
import torch
from scipy import fft
from scipy.signal import find_peaks

import fev
from sap_rpt_oss import SAP_RPT_OSS_Regressor


class SapRpt1Model(fev.ForecastingModel):
    """SAP RPT-1 applied to time series via per-item tabular regression."""

    model_name = "sap-rpt-1"

    # RPT-1 is a general tabular model, not pretrained on any time series datasets
    trained_on_datasets = []

    def __init__(self, max_context_size: int = 4096, bagging: int = 2):
        super().__init__()
        self.max_context_size = max_context_size
        self.bagging = bagging
        self._regressor = None

    @property
    def regressor(self):
        if self._regressor is None:
            self._regressor = SAP_RPT_OSS_Regressor(
                max_context_size=self.max_context_size,
                bagging=self.bagging,
            )
            self._regressor.dtype = torch.float32
            self._regressor.model = self._regressor.model.float()
        return self._regressor

    def _fit_predict(self, task: fev.Task) -> list[datasets.DatasetDict]:
        has_static = bool(getattr(task, "static_columns", None))
        is_multivariate = len(task.target_columns) > 1
        target_col = "target" if is_multivariate else task.target_columns[0]
        past_dynamic_cols = list(getattr(task, "past_dynamic_columns", []) or [])

        predictions_per_window = []

        for window in task.iter_windows():
            train_df, future_df, static_df = fev.convert_input_data(
                window, adapter="pandas", as_univariate=is_multivariate
            )

            X_train, y_train, X_future = _prepare_tabular_data(
                train_df, future_df,
                static_df=static_df if has_static else None,
                target_col=target_col,
                use_static=has_static,
                past_dynamic_cols=past_dynamic_cols,
            )

            X_train, X_future = _add_temporal_features(X_train, y_train, X_future)

            # Per-item fit/predict
            y_pred = np.full(len(X_future), np.nan)
            item_ids = X_future["id"].unique()

            for sid in item_ids:
                train_mask = X_train["id"] == sid
                future_mask = X_future["id"] == sid

                X_tr = X_train[train_mask].reset_index(drop=True)
                y_tr = y_train[train_mask].reset_index(drop=True)
                X_fu = X_future[future_mask].reset_index(drop=True)

                if len(X_tr) == 0:
                    y_pred[future_mask.values] = y_train.mean()
                    continue

                with self._record_inference_time():
                    self.regressor.fit(X_tr, y_tr)
                    preds = self.regressor.predict(X_fu)

                preds = np.where(np.isfinite(preds), preds, y_tr.mean())
                y_pred[future_mask.values] = preds

            # Fallback for remaining NaNs
            invalid = ~np.isfinite(y_pred)
            if invalid.any():
                y_pred[invalid] = y_train.mean()

            # Format predictions
            preds_fev = _format_predictions(
                y_pred, future_df, task.horizon,
                quantile_levels=task.quantile_levels,
            )

            if is_multivariate:
                predictions_per_window.append(
                    fev.combine_univariate_predictions_to_multivariate(
                        preds_fev, target_columns=task.target_columns,
                    )
                )
            else:
                predictions_per_window.append(preds_fev)

        return predictions_per_window


# ==============================================================================
# Data Processing
# ==============================================================================


def _fix_arrow_dtypes(df):
    df = df.copy()
    for col in df.columns:
        dtype_str = str(df[col].dtype)
        if "timestamp" in dtype_str.lower() or "datetime" in dtype_str.lower():
            col_dt = pd.to_datetime(df[col])
            if col_dt.dt.tz is not None:
                col_dt = col_dt.dt.tz_convert(None)
            df[col] = col_dt
        elif "[pyarrow]" in dtype_str or dtype_str == "str":
            converted = pd.to_numeric(df[col], errors="coerce")
            df[col] = converted if converted.notna().mean() > 0.5 else df[col].astype(str).astype("object")
        elif dtype_str == "float32":
            df[col] = df[col].astype("float64")
        elif "bool" in dtype_str:
            df[col] = df[col].astype(int)
    return df


def _prepare_tabular_data(train_df, future_df, static_df=None, target_col="target",
                          use_static=True, past_dynamic_cols=None):
    past_dynamic_cols = past_dynamic_cols or []

    if use_static and static_df is not None:
        train_merged = train_df.merge(static_df, on="id", how="left")
        future_merged = future_df.merge(static_df, on="id", how="left")
    else:
        train_merged = train_df.copy()
        future_merged = future_df.copy()

    # Drop past-dynamic columns
    for c in past_dynamic_cols:
        if c in train_merged.columns:
            train_merged = train_merged.drop(columns=[c])
        if c in future_merged.columns:
            future_merged = future_merged.drop(columns=[c])

    y_train = train_merged[target_col].astype("float64")
    X_train = train_merged.drop(columns=[target_col])

    valid = ~y_train.isna()
    X_train = X_train[valid].reset_index(drop=True)
    y_train = y_train[valid].reset_index(drop=True)

    X_future = future_merged.reindex(columns=X_train.columns)

    X_train = _fix_arrow_dtypes(X_train)
    X_future = _fix_arrow_dtypes(X_future)

    if "timestamp" in X_train.columns:
        X_train["timestamp"] = pd.to_datetime(X_train["timestamp"])
        X_future["timestamp"] = pd.to_datetime(X_future["timestamp"])

    return X_train, y_train, X_future


# ==============================================================================
# Temporal Feature Engineering
# ==============================================================================


def _add_temporal_features(X_train, y_train, X_future, max_seasonal=5):
    combined = pd.concat([
        X_train.assign(_is_train=True),
        X_future.assign(_is_train=False),
    ], ignore_index=True)
    combined = combined.sort_values(["id", "timestamp"])
    combined["running_index"] = combined.groupby("id").cumcount()

    X_train = combined[combined["_is_train"]].drop(columns=["_is_train"]).reset_index(drop=True)
    X_future = combined[~combined["_is_train"]].drop(columns=["_is_train"]).reset_index(drop=True)

    # Calendar features
    for df in [X_train, X_future]:
        if "timestamp" not in df.columns:
            continue
    X_train = _add_calendar(X_train)
    X_future = _add_calendar(X_future)

    # Seasonal features (FFT-based)
    X_train = X_train.copy()
    X_train["_target"] = y_train.values

    for i in range(max_seasonal):
        X_train[f"seasonal_sin_{i}"] = 0.0
        X_train[f"seasonal_cos_{i}"] = 0.0
        X_future[f"seasonal_sin_{i}"] = 0.0
        X_future[f"seasonal_cos_{i}"] = 0.0

    for item_id in X_train["id"].unique():
        tmask = X_train["id"] == item_id
        target = X_train.loc[tmask, "_target"].values
        periods = _detect_periods(target, max_top_k=max_seasonal)

        tidx = X_train.loc[tmask, "running_index"].values
        for i, p in enumerate(periods[:max_seasonal]):
            X_train.loc[tmask, f"seasonal_sin_{i}"] = np.sin(2 * np.pi * tidx / p)
            X_train.loc[tmask, f"seasonal_cos_{i}"] = np.cos(2 * np.pi * tidx / p)

        fmask = X_future["id"] == item_id
        if fmask.any():
            fidx = X_future.loc[fmask, "running_index"].values
            for i, p in enumerate(periods[:max_seasonal]):
                X_future.loc[fmask, f"seasonal_sin_{i}"] = np.sin(2 * np.pi * fidx / p)
                X_future.loc[fmask, f"seasonal_cos_{i}"] = np.cos(2 * np.pi * fidx / p)

    X_train = X_train.drop(columns=["_target"])
    return X_train, X_future


def _add_calendar(df):
    if "timestamp" not in df.columns:
        return df
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"])
    specs = [
        ("hour_of_day", ts.dt.hour, 24),
        ("day_of_week", ts.dt.dayofweek, 7),
        ("day_of_month", ts.dt.day, 30.5),
        ("day_of_year", ts.dt.dayofyear, 365),
        ("week_of_year", ts.dt.isocalendar().week.astype(int), 52),
        ("month_of_year", ts.dt.month, 12),
    ]
    for name, feat, period in specs:
        df[f"{name}_sin"] = np.sin(2 * np.pi * feat.values / max(period - 1, 1))
        df[f"{name}_cos"] = np.cos(2 * np.pi * feat.values / max(period - 1, 1))
    df["year"] = ts.dt.year
    return df


def _detect_periods(values, max_top_k=5, threshold=0.05):
    values = np.array(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) < 4:
        return []
    n = len(values)
    # Detrend
    idx = np.arange(n)
    coeffs = np.polyfit(idx, values, 1, rcond=None)
    values = values - np.polyval(coeffs, idx)
    # Hann + zero-pad + FFT
    values = values * np.hanning(n)
    padded = np.zeros(n * 2)
    padded[:n] = values
    mag = np.abs(fft.rfft(padded))
    freqs = np.fft.rfftfreq(n * 2, d=1.0)
    mag[0] = 0.0
    peaks, _ = find_peaks(mag, height=threshold * mag.max())
    if len(peaks) == 0:
        peaks = np.arange(1, len(mag))
    top = peaks[np.argsort(mag[peaks])[::-1]][:max_top_k]
    periods = []
    for i in top:
        if freqs[i] > 0:
            p = round(1.0 / freqs[i])
            if p > 0 and p not in periods:
                periods.append(p)
    return periods[:max_top_k]


# ==============================================================================
# Prediction Formatting
# ==============================================================================


def _format_predictions(y_pred, future_df, horizon, quantile_levels=None):
    future_df = future_df.copy()
    future_df["pred"] = y_pred
    rows = []
    for sid in sorted(future_df["id"].unique()):
        preds = future_df[future_df["id"] == sid].sort_values("timestamp")["pred"].tolist()
        assert len(preds) == horizon
        row = {"predictions": preds}
        if quantile_levels:
            for q in quantile_levels:
                row[str(q)] = preds
        rows.append(row)
    return rows
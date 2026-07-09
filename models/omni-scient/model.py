"""Champion-challenger forecasting ensemble.

A deterministic, pure-stdlib ensemble that segments each time series (stable,
seasonal, lumpy, volatile, new) and runs four candidate models per series:
SNAIVE (seasonal naive), AVG3 (trailing average), SIDX (seasonal index x trend),
and HW (Holt-Winters additive). The champion is picked per series by WAPE on a
held-out validation window, then retrained on the full history for the forecast.
"""

from __future__ import annotations

import math
import statistics
from typing import Any

import datasets
import numpy as np
import pandas as pd

import fev


class OmniScientModel(fev.ForecastingModel):
    """Champion-challenger ensemble forecaster — deterministic, pure stdlib."""

    model_name = "omni-scient"
    trained_on_datasets: list[str] = []  # trains from scratch on each task

    def __init__(
        self,
        include_quantiles: bool = True,
        max_context_length: int | None = 5_000,
    ):
        super().__init__()
        self.include_quantiles = include_quantiles
        self.max_context_length = max_context_length

    # ------------------------------------------------------------------
    # fev entry point
    # ------------------------------------------------------------------

    def _fit_predict(self, task: fev.Task) -> list[datasets.DatasetDict]:
        predictions_per_window: list[datasets.DatasetDict] = []
        for window in task.iter_windows():
            preds = self._predict_window(window, task.quantile_levels)
            predictions_per_window.append(preds)
        return predictions_per_window

    def _predict_window(
        self,
        window: fev.EvaluationWindow,
        quantile_levels: list[float],
    ) -> datasets.DatasetDict:
        # Convert to pandas long format, splitting multivariate into univariate
        past_df, future_df, _static_df = fev.convert_input_data(
            window, adapter="pandas", as_univariate=True
        )

        id_col = window.id_column
        target_col = "target"  # as_univariate=True renames to "target"
        horizon = window.horizon

        # Extract per-series historical values (ordered as returned by adapter)
        train_series: list[list[float]] = []
        for _sid, grp in past_df.groupby(id_col, sort=False):
            vals = grp[target_col].tolist()
            if self.max_context_length and len(vals) > self.max_context_length:
                vals = vals[-self.max_context_length:]
            train_series.append(vals)

        n_series = len(train_series)

        # Forecast each series
        forecast_rows: list[dict[str, Any]] = []
        with self._record_inference_time():
            for i in range(n_series):
                train = train_series[i]
                fc = _gm_forecast(train, horizon)
                row: dict[str, Any] = {"predictions": fc}
                if quantile_levels and self.include_quantiles:
                    sigmas = _estimate_quantile_widths(train, horizon)
                    for q in quantile_levels:
                        z = _normal_quantile(q)
                        row[str(q)] = [fc[k] + z * sigmas[k] for k in range(horizon)]
                else:
                    for q in quantile_levels:
                        row[str(q)] = [0.0] * horizon
                forecast_rows.append(row)

        # Build predictions DataFrame for convert_forecast_df_to_predictions
        flat_rows = []
        for row in forecast_rows:
            for k in range(horizon):
                flat_row = {col: row[col][k] for col in row}
                flat_rows.append(flat_row)
        forecast_df = pd.DataFrame(flat_rows)

        return fev.utils.convert_forecast_df_to_predictions(
            forecast_df,
            horizon=horizon,
            quantile_levels=quantile_levels,
            target_columns=window.target_columns,
        )


# ------------------------------------------------------------------
# GM Modular engine — pure stdlib, deterministic
# ------------------------------------------------------------------

_SNAIVE = "SNAIVE"
_AVG3 = "AVG3"
_SIDX = "SIDX"
_HW = "HW"
_CANDIDATES = (_SNAIVE, _AVG3, _SIDX, _HW)


def _gm_forecast(train: list[float], horizon: int) -> list[float]:
    """Run the full champion-challenger pipeline on a single series."""
    clean = [0.0 if math.isnan(v) or math.isinf(v) else v for v in train]
    if not clean or all(v == 0 for v in clean):
        return [0.0] * horizon

    seg = _segment(clean)
    val_len = max(1, min(horizon, len(clean) // 3))
    if val_len > len(clean):
        val_len = 1
    train_inner = clean[:-val_len]
    actual = clean[-val_len:]

    if not train_inner or len(train_inner) < 2:
        base = _safe_mean(clean[-3:]) if clean else 0.0
        return [base] * horizon

    scores: dict[str, float] = {}
    for cand in _CANDIDATES:
        fc = _run_candidate(cand, train_inner, val_len)
        w = _wape(actual, fc)
        if w is not None and not math.isnan(w):
            scores[cand] = w

    champion = min(scores, key=lambda k: scores[k]) if scores else _AVG3
    return _run_candidate(champion, clean, horizon)


# ---- helpers ---------------------------------------------------------------


def _sanitize(values: list[float]) -> list[float]:
    return [0.0 if math.isnan(v) or math.isinf(v) else v for v in values]


def _safe_mean(values: list[float]) -> float:
    clean = _sanitize(values)
    return statistics.mean(clean) if clean else 0.0


def _safe_pstdev(values: list[float]) -> float:
    clean = _sanitize(values)
    if len(clean) < 2:
        return 0.0
    return statistics.pstdev(clean)


# ---- segmentation ---------------------------------------------------------


def _segment(y: list[float]) -> str:
    clean = _sanitize(y)
    nz = [v for v in clean if v > 0]
    if len(nz) < 4:
        return "NEW"
    zero_share = 1 - len(nz) / len(clean)
    mean_y = _safe_mean(clean)
    cv = (_safe_pstdev(clean) / mean_y) if mean_y else 9.0
    if zero_share > 0.25:
        return "LUMPY"
    if _seasonality_strength(clean) > 0.18:
        return "SEASONAL"
    return "STABLE" if cv < 0.35 else "VOLATILE"


def _seasonality_strength(y: list[float], period: int = 12) -> float:
    clean = _sanitize(y)
    if sum(clean) == 0:
        return 0.0
    n = len(clean)
    xm = (n - 1) / 2
    mu = _safe_mean(clean)
    sxx = sum((i - xm) ** 2 for i in range(n)) or 1
    b = sum((i - xm) * (v - mu) for i, v in enumerate(clean)) / sxx
    det = [max(0.0, v - b * (i - xm)) for i, v in enumerate(clean)]
    mu2 = _safe_mean(det)
    if mu2 <= 0:
        return 0.0
    by_mod = {}
    for i, v in enumerate(det):
        by_mod.setdefault(i % period, []).append(v)
    idx = [_safe_mean(vs) / mu2 for vs in by_mod.values()]
    return _safe_pstdev(idx)


# ---- candidate models -----------------------------------------------------


def _run_candidate(name: str, train: list[float], horizon: int) -> list[float]:
    if name == _SNAIVE:
        return _f_snaive(train, horizon)
    if name == _AVG3:
        return _f_avg3(train, horizon)
    if name == _SIDX:
        return _f_sidx(train, horizon)
    return _f_hw(train, horizon)


def _f_snaive(train: list[float], horizon: int) -> list[float]:
    period = min(12, len(train))
    if len(train) < period:
        base = statistics.mean(train[-3:]) if train else 0.0
        return [base] * horizon
    out = []
    for k in range(horizon):
        idx = len(train) - period + (k % period)
        if idx >= 0:
            out.append(train[idx])
        else:
            out.append(statistics.mean(train[-3:]))
    return out


def _f_avg3(train: list[float], horizon: int) -> list[float]:
    if not train:
        return [0.0] * horizon
    base = statistics.mean(train[-3:])
    return [base] * horizon


def _f_sidx(train: list[float], horizon: int, period: int = 12) -> list[float]:
    if len(train) < period + 1 or sum(train) == 0:
        return _f_avg3(train, horizon)
    mu = statistics.mean(train)
    by_mod: dict[int, list[float]] = {}
    for i, v in enumerate(train):
        by_mod.setdefault(i % period, []).append(v)
    idx = {
        m: (statistics.mean(vs) / mu if mu else 1.0) or 1.0
        for m, vs in by_mod.items()
    }
    des = [v / idx.get(i % period, 1.0) for i, v in enumerate(train)]
    n = len(des)
    xm = (n - 1) / 2
    ym_ = statistics.mean(des)
    sxy = sum((i - xm) * (v - ym_) for i, v in enumerate(des))
    sxx = sum((i - xm) ** 2 for i in range(n)) or 1
    b = sxy / sxx
    a = ym_ - b * xm
    out = []
    for k in range(horizon):
        mth = (len(train) + k) % period
        out.append(max(0.0, (a + b * (n + k)) * idx.get(mth, 1.0)))
    return out


def _f_hw(train: list[float], horizon: int, period: int = 12) -> list[float]:
    m = min(period, len(train) // 2) if len(train) >= 4 else 1
    if len(train) < m + 3 or sum(train) == 0:
        return _f_sidx(train, horizon, period)

    best, best_err = (0.2, 0.05, 0.1), None
    for alpha in (0.2, 0.4):
        for gamma in (0.1, 0.3):
            beta = 0.05
            lvl = statistics.mean(train[:m])
            trd = (
                (statistics.mean(train[m : 2 * m]) - lvl) / m
                if len(train) >= 2 * m
                else 0.0
            )
            seas = [train[i] - lvl for i in range(m)]
            err = 0.0
            L, T, Sn = lvl, trd, list(seas)
            for i in range(m, len(train)):
                pred = L + T + Sn[i % m]
                err += abs(train[i] - pred)
                Ln = alpha * (train[i] - Sn[i % m]) + (1 - alpha) * (L + T)
                T = beta * (Ln - L) + (1 - beta) * T
                Sn[i % m] = gamma * (train[i] - Ln) + (1 - gamma) * Sn[i % m]
                L = Ln
            if best_err is None or err < best_err:
                best_err, best = err, (alpha, beta, gamma)

    alpha, beta, gamma = best
    L = statistics.mean(train[:m])
    T = (
        (statistics.mean(train[m : 2 * m]) - L) / m
        if len(train) >= 2 * m
        else 0.0
    )
    Sn = [train[i] - L for i in range(m)]
    for i in range(m, len(train)):
        Ln = alpha * (train[i] - Sn[i % m]) + (1 - alpha) * (L + T)
        T = beta * (Ln - L) + (1 - beta) * T
        Sn[i % m] = gamma * (train[i] - Ln) + (1 - gamma) * Sn[i % m]
        L = Ln
    n = len(train)
    return [max(0.0, L + (k + 1) * T + Sn[(n + k) % m]) for k in range(horizon)]


# ---- metrics --------------------------------------------------------------


def _wape(actual: list[float], forecast: list[float]) -> float | None:
    den = sum(abs(a) for a in actual)
    if den == 0:
        return None
    return sum(abs(a - f) for a, f in zip(actual, forecast)) / den


# ---- quantile helpers -----------------------------------------------------


def _estimate_quantile_widths(
    train: list[float], horizon: int, period: int = 12
) -> np.ndarray:
    clean = _sanitize(train)
    if len(clean) < period + 2:
        sd = _safe_pstdev(clean) if len(clean) > 1 else 1.0
        return np.full(horizon, sd, dtype=np.float64)

    mu = _safe_mean(clean) or 1.0
    by_mod: dict[int, list[float]] = {}
    for i, v in enumerate(clean):
        by_mod.setdefault(i % period, []).append(v)
    idx = {m: (_safe_mean(vs) / mu) or 1.0 for m, vs in by_mod.items()}
    residuals = []
    for i in range(1, len(clean)):
        pred = clean[i - 1] * (
            idx.get(i % period, 1.0) / idx.get((i - 1) % period, 1.0)
        )
        residuals.append(clean[i] - pred)
    sd = (
        _safe_pstdev(residuals)
        if len(residuals) > 1
        else (_safe_pstdev(clean) if len(clean) > 1 else 1.0)
    )
    return np.array(
        [sd * math.sqrt(k + 1) for k in range(horizon)], dtype=np.float64
    )


def _normal_quantile(q: float) -> float:
    """Rational approximation of the standard normal quantile (Abramowitz & Stegun)."""
    if q <= 0 or q >= 1:
        return 0.0
    p = q if q <= 0.5 else 1 - q
    t = math.sqrt(-2.0 * math.log(p))
    c = [2.515517, 0.802853, 0.010328]
    d = [1.432788, 0.189269, 0.001308]
    num = c[0] + c[1] * t + c[2] * t * t
    den = 1.0 + d[0] * t + d[1] * t * t + d[2] * t * t * t
    z = t - num / den
    return -z if q <= 0.5 else z

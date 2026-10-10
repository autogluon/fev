import numpy as np

def shrunk_log_slope(history, period=52, var_inflation=4.0, cap=0.5):
    y = np.asarray(history, dtype=float)
    L = len(y)
    if L < 8:
        return 0.0
    eps = 0.05 * max(float(np.mean(np.abs(y))), 1e-12)
    ly = np.log(np.maximum(y, 0.0) + eps)
    t = (np.arange(L) - (L - 1) / 2.0) / float(period)
    A = np.vstack([np.ones(L), t]).T
    coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
    resid = ly - A @ coef
    s2 = float(np.sum(resid ** 2)) / max(L - 2, 1)
    denom = float(np.sum(t ** 2))
    if denom <= 0:
        return 0.0
    se2 = var_inflation * s2 / denom
    b = float(coef[1])
    shrink = b * b / (b * b + se2) if b * b + se2 > 0 else 0.0
    return float(np.clip(b * shrink, -cap, cap))

def growth_path(slope, horizon, period=52, horizon_damp=0.75):
    h = np.arange(1, horizon + 1) / float(period)
    return np.exp(slope * h * horizon_damp)

def fitted_level(history, slope, total_length, period=52):
    y = np.asarray(history, dtype=float)
    L = len(y)
    eps = 0.05 * max(float(np.mean(np.abs(y))), 1e-12)
    ly = np.log(np.maximum(y, 0.0) + eps)
    t = (np.arange(L) - (L - 1) / 2.0) / float(period)
    a = float(np.mean(ly) - slope * np.mean(t))
    tt = (np.arange(total_length) - (L - 1) / 2.0) / float(period)
    return np.maximum(np.exp(a + slope * tt) - eps, 0.0)

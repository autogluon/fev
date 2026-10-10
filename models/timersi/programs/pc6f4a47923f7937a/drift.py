import numpy as np
WEIGHT = 0.3
MAX_STEP_ADJ = 0.08

def own_drift(history, observed=None):
    y = np.asarray(history, float)
    m = np.isfinite(y)
    if observed is not None:
        m &= np.asarray(observed, bool)
    y = y[m]
    if y.size < 2 or not np.all(y > 0):
        return np.nan
    return float(np.median(np.diff(np.log(y))))

def adjust_median(median_path, last_value, history, observed=None, weight=WEIGHT, cap=MAX_STEP_ADJ):
    med = np.asarray(median_path, float)
    H = med.size
    g_own = own_drift(history, observed)
    if not np.isfinite(g_own) or not np.isfinite(last_value) or last_value <= 0 or np.any(~np.isfinite(med)) or np.any(med <= 0):
        return med
    g_nat = (np.log(med[-1]) - np.log(last_value)) / H
    delta = np.clip(weight * (g_own - g_nat), -cap, cap)
    h = np.arange(1, H + 1, dtype=float)
    return med * np.exp(delta * h)

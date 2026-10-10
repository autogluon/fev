import numpy as np

def weekly_log_growth(history, window=21, season=7):
    h = np.asarray(history, float)
    h = h[np.isfinite(h)]
    if h.size < season + 4:
        return 0.0
    ma = np.convolve(h, np.ones(season) / season, 'valid')
    y = np.log(ma[-window:] + 0.5)
    n = y.size
    if n < 4:
        return 0.0
    i, j = np.triu_indices(n, 1)
    slopes = (y[j] - y[i]) / (j - i)
    return float(np.median(slopes)) * season

def trend_factor(growth, horizon, damping, cap, season=7):
    g = float(np.clip(growth, -cap, cap))
    weeks = np.arange(1, horizon + 1) / float(season)
    return np.exp(damping * g * weeks)

import numpy as np

def circular_smooth(base, half_window):
    p = len(base)
    if half_window <= 0 or p < 3:
        return np.asarray(base, dtype=float).copy()
    k = int(min(half_window, (p - 1) // 2))
    ext = np.concatenate([base[-k:], base, base[:k]])
    ker = np.ones(2 * k + 1) / (2 * k + 1)
    return np.convolve(ext, ker, mode='valid')

def seasonal_index(history, total_length, period=52, half_window=3):
    y = np.asarray(history, dtype=float)
    L = len(y)
    if L < period + 1:
        level = float(np.mean(y[-min(L, 13):])) if L else 0.0
        return (np.full(total_length, level), False)
    start = L - period
    base = y[start:L]
    prof = circular_smooth(base, half_window)
    idx = (np.arange(total_length) - start) % period
    return (prof[idx], True)

def noise_ratio(history):
    y = np.asarray(history, dtype=float)
    if len(y) < 3:
        return 1.0
    lvl = max(float(np.mean(np.abs(y))), 1e-12)
    return float(np.mean(np.abs(np.diff(y))) / lvl)

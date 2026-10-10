import numpy as np

def mad_scale(x, eps=1e-09):
    x = np.asarray(x, float)
    med = np.median(x)
    return float(max(1.4826 * np.median(np.abs(x - med)), eps))

def seasonal_scale(x, period=7, eps=1e-08):
    x = np.asarray(x, float)
    if x.size > period:
        v = float(np.abs(x[period:] - x[:-period]).mean())
    else:
        v = float(np.abs(np.diff(x)).mean()) if x.size > 1 else 0.0
    if not np.isfinite(v) or v <= 0:
        v = mad_scale(x)
    return max(v, eps)

def theil_sen(y, max_pairs=4000, rng_seed=0):
    y = np.asarray(y, float)
    n = y.size
    if n < 3:
        return 0.0
    i, j = np.triu_indices(n, 1)
    if i.size > max_pairs:
        rs = np.random.RandomState(rng_seed)
        sel = rs.choice(i.size, max_pairs, replace=False)
        i, j = (i[sel], j[sel])
    slopes = (y[j] - y[i]) / (j - i)
    slopes = slopes[np.isfinite(slopes)]
    return float(np.median(slopes)) if slopes.size else 0.0

def robust_level(x, window=7):
    x = np.asarray(x, float)
    return float(np.median(x[-min(window, x.size):]))

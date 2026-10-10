import numpy as np

def _rolling_median(a, w):
    n = a.shape[-1]
    half = w // 2
    out = np.empty_like(a)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[..., i] = np.median(a[..., lo:hi], axis=-1)
    return out

def slow_trend(X, seas=48, smooth_days=7, flat_tail_days=3):
    D, L = X.shape
    nd = L // seas
    if nd < 4 * smooth_days:
        return np.zeros_like(X)
    head = L - nd * seas
    m = np.median(X[:, head:].reshape(D, nd, seas), axis=2)
    t = _rolling_median(m, smooth_days)
    if flat_tail_days > 0:
        t[:, -flat_tail_days:] = t[:, -flat_tail_days - 1:-flat_tail_days]
    centres = head + np.arange(nd) * seas + seas // 2
    grid = np.arange(L)
    out = np.empty((D, L))
    for i in range(D):
        out[i] = np.interp(grid, centres, t[i])
    return out

def detrend(X, seas=48, smooth_days=7, flat_tail_days=3, strength=1.0):
    M = slow_trend(X, seas, smooth_days, flat_tail_days)
    adj = strength * (M - M[:, -1:])
    return (X - adj, adj)

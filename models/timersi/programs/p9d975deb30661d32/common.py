import numpy as np
SEASON = 96
WEEK = 96 * 7

def finite(a, fill=0.0):
    a = np.asarray(a, dtype=np.float64)
    if not np.all(np.isfinite(a)):
        a = np.where(np.isfinite(a), a, fill)
    return a

def history(view):
    h = np.asarray(view['target_history'], dtype=np.float64)
    obs = np.asarray(view['target_observed'])
    if obs.shape == h.shape:
        h = np.where(obs.astype(bool), h, np.nan)
    out = np.empty_like(h)
    for d in range(h.shape[0]):
        row = h[d]
        idx = np.where(np.isfinite(row))[0]
        if idx.size == 0:
            out[d] = 0.0
            continue
        filled = row.copy()
        bad = ~np.isfinite(filled)
        if bad.any():
            last = np.where(np.isfinite(filled), np.arange(filled.size), -1)
            np.maximum.accumulate(last, out=last)
            first = idx[0]
            last[last < 0] = first
            filled = filled[last]
        out[d] = filled
    return out

def seasonal_scale(past):
    if past.shape[1] <= SEASON:
        s = np.abs(np.diff(past, axis=1)).mean(axis=1)
    else:
        s = np.abs(past[:, SEASON:] - past[:, :-SEASON]).mean(axis=1)
    s = np.asarray(s, dtype=np.float64)
    med = np.median(s[s > 0]) if np.any(s > 0) else 1.0
    return np.where(s > 1e-09, s, max(med, 1e-09))

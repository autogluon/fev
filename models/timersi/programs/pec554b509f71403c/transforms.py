import numpy as np
MIN_POSITIVE_FRACTION = 1.0
MIN_DYNAMIC_RANGE = 1.5
MIN_POINTS = 24

def _observed(hist, mask):
    v = hist[mask] if mask is not None else hist
    return v[np.isfinite(v)]

def plan_log(hist, mask):
    v = _observed(hist, mask)
    if v.size < MIN_POINTS:
        return False
    if not np.all(v > 0):
        return False
    lo, hi = (float(np.min(v)), float(np.max(v)))
    if lo <= 0 or hi / lo < MIN_DYNAMIC_RANGE:
        return False
    return True

def forward(hist, use_log):
    out = np.array(hist, dtype=float)
    if use_log:
        out = np.log(np.maximum(out, 1e-09))
    return out

def inverse(values, use_log):
    if not use_log:
        return values
    return np.exp(np.clip(values, -700.0, 700.0))

import numpy as np

def _beta_single(y, horizon, k, minhist):
    y = np.asarray(y, float)
    n = len(y)
    if n < minhist + 2:
        return None
    num = 0.0
    den = 0.0
    for t in range(minhist, n - 1):
        lo = max(1, t - k + 1)
        drift = float(np.mean(np.diff(y[lo - 1:t + 1])))
        for h in range(1, min(horizon, n - 1 - t) + 1):
            dhat = h * drift
            num += (y[t + h] - y[t]) * dhat
            den += dhat * dhat
    if den <= 0.0:
        return None
    return num / den

def drift_realisation_beta(y, horizon=8, windows=(2, 4, 6), minhist=8):
    vals = [b for k in windows for b in [_beta_single(y, horizon, k, minhist)] if b is not None]
    if not vals:
        return None
    return float(np.median(vals))

def damping_lambda(y, horizon=8, intercept=0.5, slope=0.8, floor=0.4, cap=1.0, windows=(2, 4, 6), minhist=8):
    beta = drift_realisation_beta(y, horizon=horizon, windows=windows, minhist=minhist)
    if beta is None or not np.isfinite(beta):
        return (1.0, None)
    return (float(np.clip(intercept + slope * beta, floor, cap)), beta)

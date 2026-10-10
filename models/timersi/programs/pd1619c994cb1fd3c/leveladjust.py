import numpy as np
ALPHA = 0.4
RECENT = 3
LOCAL = 8
CAP_IN_SCALES = 1.5

def local_level_shift(history, alpha=ALPHA, recent=RECENT, local=LOCAL, cap_in_scales=CAP_IN_SCALES):
    y = np.asarray(history, dtype=float)
    D, L = y.shape
    shift = np.zeros(D)
    if L < 4:
        return shift
    k = min(recent, L)
    m = min(local, L)
    w = np.arange(1, k + 1, dtype=float)
    for i in range(D):
        series = y[i]
        rec = float(np.average(series[-k:], weights=w))
        loc = float(series[-m:].mean())
        scale = float(np.abs(np.diff(series)).mean()) if L > 1 else 0.0
        s = alpha * (rec - loc)
        if scale > 0:
            s = float(np.clip(s, -cap_in_scales * scale, cap_in_scales * scale))
        shift[i] = s
    return shift
BETA = 0.2
PHI = 0.6
TREND_WINDOW = 6
TREND_CAP_IN_SCALES = 1.0

def damped_trend(history, horizon, beta=BETA, phi=PHI, window=TREND_WINDOW, cap_in_scales=TREND_CAP_IN_SCALES):
    y = np.asarray(history, dtype=float)
    D, L = y.shape
    h = np.arange(1, int(horizon) + 1, dtype=float)
    damp = np.cumsum(phi ** h)
    out = np.zeros((D, int(horizon)))
    if L < 4:
        return out
    k = min(window, L)
    x = np.arange(k, dtype=float) - (k - 1) / 2.0
    denom = float((x * x).sum())
    for i in range(D):
        seg = y[i][-k:]
        slope = float((x * (seg - seg.mean())).sum() / denom) if denom > 0 else 0.0
        scale = float(np.abs(np.diff(y[i])).mean()) if L > 1 else 0.0
        if scale > 0:
            slope = float(np.clip(slope, -cap_in_scales * scale, cap_in_scales * scale))
        out[i] = beta * slope * damp
    return out

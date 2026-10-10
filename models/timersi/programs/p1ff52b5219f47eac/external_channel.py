import numpy as np
SMOOTH = 13
MAX_RATIO = 5.0

def _smooth(x, w):
    w = int(max(1, w | 1))
    k = w // 2
    ext = np.concatenate([np.full(k, x[:k + 1].mean()), x, np.full(k, x[-(k + 1):].mean())])
    return np.convolve(ext, np.ones(w) / w, mode='valid')[:len(x)]

def level_channel(ext_full, y_past, Lc):
    e = np.asarray(ext_full, float)
    if e.size == 0 or not np.isfinite(e).all():
        return None
    es = _smooth(e, SMOOTH)
    ep = es[:Lc]
    y = np.asarray(y_past, float)
    den = float(np.mean(np.abs(ep)))
    num = float(np.mean(np.abs(y)))
    if den <= 1e-12 or num <= 1e-12:
        return None
    out = es * (num / den)
    hi = MAX_RATIO * max(float(np.max(np.abs(y))), 1e-12)
    return np.clip(out, -hi, hi)

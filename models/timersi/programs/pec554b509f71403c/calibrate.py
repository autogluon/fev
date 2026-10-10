import numpy as np
BASE = 0.85
EXPONENT = 0.4
CLIP_LOW = 0.6
CLIP_HIGH = 1.15
_H = 12
_ORIGINS = 20
_STEP = 6
_DRIFT_WIN = 24
_PHI = 0.85
_PCT = 80
_MIN_PAST = 60

def surrogate_band(history, horizon=_H):
    h = np.asarray(history, dtype=float)
    errs = []
    for j in range(_ORIGINS):
        T = len(h) - horizon - j * _STEP
        if T < _MIN_PAST:
            break
        past, fut = (h[:T], h[T:T + horizon])
        w = min(_DRIFT_WIN, T - 1)
        d = (past[-1] - past[-1 - w]) / float(w)
        cum = np.cumsum([_PHI ** t for t in range(1, horizon + 1)])
        errs.append(np.abs(fut - (past[-1] + d * cum)))
    if not errs:
        return None
    return float(np.percentile(np.concatenate(errs), _PCT) * 2.0)

def spread_factor(history, quantiles):
    q = np.asarray(quantiles, dtype=float)
    width = float(np.mean(q[..., -1] - q[..., 0]))
    band = surrogate_band(history)
    if band is None or not np.isfinite(band) or width <= 0:
        return BASE
    ratio = band / width
    if not np.isfinite(ratio) or ratio <= 0:
        return BASE
    return float(np.clip(BASE * ratio ** EXPONENT, CLIP_LOW, CLIP_HIGH))

def rescale(quantiles, factor):
    q = np.asarray(quantiles, dtype=float)
    med = q[..., 4:5]
    return np.sort(med + factor * (q - med), axis=-1)

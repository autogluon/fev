import numpy as np
from calendar_ops import parse_day, week_features
QL = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
GRID_W = (0.0, 0.15, 0.3, 0.45)
SHRINK = 0.5
W_CAP = 0.35
MIN_FIT = 30
MIN_ANNUAL = 57
RIDGE_BASE = 2.0

def _design(ts):
    Xw = week_features([str(t) for t in ts])
    days = [parse_day(t) for t in ts]
    doy = np.array([(d - d.replace(month=1, day=1)).days / 365.25 for d in days])
    T = len(ts)
    t_lin = np.arange(T, dtype=float)
    cols = [Xw[:, 0], Xw[:, 1], Xw[:, 2], Xw[:, 3], Xw[:, 4], Xw[:, 5], np.cos(2 * np.pi * doy), np.sin(2 * np.pi * doy), np.cos(4 * np.pi * doy), np.sin(4 * np.pi * doy)]
    return (np.stack(cols, 1), t_lin)

def fit_predict(y, obs, ts_past, ts_future):
    y = np.asarray(y, float)
    keep = np.asarray(obs, bool) & np.isfinite(y) & (y > 0)
    if keep.sum() < MIN_FIT:
        return None
    Xp, tp = _design(ts_past)
    Xf, _ = _design(ts_future)
    if keep.sum() < MIN_ANNUAL:
        Xp = Xp[:, :6]
        Xf = Xf[:, :6]
    tf = np.arange(len(ts_past), len(ts_past) + len(ts_future), dtype=float)
    ly = np.log(y[keep])
    Xo = Xp[keep]
    to = tp[keep]
    tmu, tsd = (to.mean(), max(to.std(), 1.0))
    A = np.c_[np.ones(keep.sum()), (to - tmu) / tsd, Xo]
    Af = np.c_[np.ones(len(tf)), (tf - tmu) / tsd, Xf]
    pen = np.r_[0.0, 0.5, np.full(Xo.shape[1], RIDGE_BASE)]
    try:
        beta = np.linalg.solve(A.T @ A + np.diag(pen), A.T @ ly)
    except np.linalg.LinAlgError:
        return None
    fore = Af @ beta
    if not np.all(np.isfinite(fore)):
        return None
    return np.clip(fore, np.log(max(y[keep].min(), 0.001)) - 1.0, np.log(y[keep].max()) + 1.0)

def _pinball(y, q, scale):
    d = y[:, None] - q
    return float(np.maximum(QL * d, (QL - 1) * d).mean() * 2.0 / scale)

def blend(point, quant, spec_log, w):
    p = np.asarray(point, float)
    lp = np.log(np.maximum(p, 1e-06))
    shift = np.exp(w * np.clip(spec_log - lp, -0.3, 0.3))
    return (p * shift, np.asarray(quant, float) * shift[:, None])

def choose_weight(replays):
    info = {'n_replays': len(replays)}
    if not replays:
        info['reason'] = 'no matured replay with a fitted specialist'
        return (0.0, info)
    curve = {}
    for w in GRID_W:
        vals = []
        for y, pc, qc, sl, sc in replays:
            _, qb = blend(pc, qc, sl, w)
            vals.append(_pinball(y, qb, sc))
        curve[w] = float(np.mean(vals))
    best = min(curve, key=curve.get)
    w = float(np.clip(SHRINK * best, 0.0, W_CAP))
    info.update({'curve': {str(k): round(v, 5) for k, v in curve.items()}, 'argmin': best, 'applied': w})
    return (w, info)

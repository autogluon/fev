import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from .features import build_matrix
except Exception:
    from features import build_matrix
PARAMS = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=40, subsample=0.9, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=4, random_state=0)
ANCHOR_LAGS = (168, 336)

def _model():
    import lightgbm as lgb
    return lgb.LGBMRegressor(**PARAMS)

def _subset(X, names, drop):
    if not drop:
        return X
    keep = [i for i, n in enumerate(names) if n not in drop]
    return np.ascontiguousarray(X[:, keep])

def covariate_estimate(ts, temp, y_past, horizon, n_blocks=4, oos_tail=8760):
    X, _ = build_matrix(np.asarray(ts, dtype='datetime64[h]'), temp)
    L = len(y_past)
    Xp, Xf = (X[:L], X[L:L + horizon])
    y = np.asarray(y_past, float)
    ok = np.isfinite(y)
    m = _model()
    m.fit(Xp[ok], y[ok])
    future = m.predict(Xf)
    oos = np.full(L, np.nan)
    if n_blocks and n_blocks > 1:
        start = 0 if oos_tail is None else max(0, L - oos_tail)
        edges = np.linspace(start, L, n_blocks + 1).astype(int)
        for b in range(n_blocks):
            lo, hi = (edges[b], edges[b + 1])
            if hi <= lo:
                continue
            mask = np.ones(L, bool)
            mask[lo:hi] = False
            mask &= ok
            mm = _model()
            mm.fit(Xp[mask], y[mask])
            oos[lo:hi] = mm.predict(Xp[lo:hi])
    return (future, oos)

def absolute_notrend(ts, temp, y_past, horizon):
    X, names = build_matrix(np.asarray(ts, dtype='datetime64[h]'), temp)
    X = _subset(X, names, {'trend'})
    L = len(y_past)
    y = np.asarray(y_past, float)
    ok = np.isfinite(y)
    m = _model()
    m.fit(X[:L][ok], y[ok])
    return m.predict(X[L:L + horizon])

def _lag_mean(series, n, lags):
    cols = []
    for lg in lags:
        c = np.full(n, np.nan)
        if lg < n:
            c[lg:] = series[:n - lg]
        cols.append(c)
    stack = np.array(cols)
    with np.errstate(invalid='ignore'):
        out = np.nanmean(np.where(np.isfinite(stack), stack, np.nan), axis=0)
    return out

def deviation(ts, temp, y_past, horizon, lags=ANCHOR_LAGS):
    ts = np.asarray(ts, dtype='datetime64[h]')
    X, names = build_matrix(ts, temp)
    X = _subset(X, names, {'trend'})
    L = len(y_past)
    n = L + horizon
    y = np.asarray(y_past, float)
    ypad = np.full(n, np.nan)
    ypad[:L] = y
    anchor = _lag_mean(ypad, n, lags)
    T = np.asarray(temp, float)[:n]
    dT = T - _lag_mean(T, n, lags)
    dT = np.where(np.isfinite(dT), dT, 0.0)
    X = np.column_stack([X, dT, np.maximum(dT, 0.0), np.minimum(dT, 0.0)])
    tgt = y - anchor[:L]
    ok = np.isfinite(tgt) & np.isfinite(X[:L]).all(axis=1)
    if ok.sum() < 500:
        return None
    m = _model()
    m.fit(X[:L][ok], tgt[ok])
    out = m.predict(X[L:n]) + anchor[L:n]
    return out if np.isfinite(out).all() else None

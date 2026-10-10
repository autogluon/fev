import numpy as np
PARAMS = dict(n_estimators=400, learning_rate=0.04, num_leaves=63, min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=4, verbosity=-1)
TRAIN_DAYS = 1092

def _mk(X, rows):
    F = X.shape[2]
    return np.concatenate([X[rows].reshape(len(rows) * 24, F), np.tile(np.arange(24), len(rows))[:, None]], 1)

def _fit_predict(X, P, fit_rows, pred_rows):
    import lightgbm as lgb
    blk = P[fit_rows][np.isfinite(P[fit_rows])]
    med = np.median(blk)
    mad = np.median(np.abs(blk - med)) * 1.4826 + 1e-06
    Xtr = _mk(X, fit_rows)
    ytr = np.arcsinh((P[fit_rows].reshape(-1) - med) / mad)
    ok = np.isfinite(ytr) & np.isfinite(Xtr).all(1)
    if ok.sum() < 2000:
        return None
    m = lgb.LGBMRegressor(**PARAMS)
    m.fit(Xtr[ok], ytr[ok], categorical_feature=[Xtr.shape[1] - 1])
    Xte = _mk(X, pred_rows)
    out = np.sinh(m.predict(np.nan_to_num(Xte, nan=0.0))) * mad + med
    return out.reshape(len(pred_rows), 24)

def gbm_forecast(X, P, origin, oos=3):
    start = max(16, origin - TRAIN_DAYS)
    fit_rows = np.arange(start, origin)
    if len(fit_rows) < 120:
        return (None, None)
    today = _fit_predict(X, P, fit_rows, np.arange(origin, origin + 1))
    today = None if today is None else today[0]
    hist = None
    if oos > 0 and origin - oos > start + 120:
        hist = _fit_predict(X, P, np.arange(start, origin - oos), np.arange(origin - oos, origin))
    return (today, hist)

import numpy as np
import calendar_features as cf
FEATURES = ['tod', 'daytype', 'dow', 'temp', 'temp24', 'temp72', 'hdd24', 'cdd24', 'hdd', 'cdd', 'rad_diffuse', 'rad_direct', 'sin_year', 'cos_year', 'sin2_year', 'cos2_year', 'trend']
CATEGORICAL = [0, 1, 2]

def _trailing_mean(x, w):
    c = np.convolve(np.asarray(x, dtype=np.float64), np.ones(w) / w, mode='full')
    return c[:len(x)]

def design(item_id, unix, temp, rdif, rdir):
    dt, tod, days = cf.day_type(item_id, unix)
    y, m, d, _, dow, _ = cf.civil_fields(unix)
    jan1 = np.array([cf._ymd_to_unix(int(yy), 1, 1) // 86400 for yy in y], dtype=np.int64)
    ang = 2 * np.pi * (days - jan1) / 365.25
    t24, t72 = (_trailing_mean(temp, 48), _trailing_mean(temp, 144))
    return np.column_stack([tod, dt, dow, temp, t24, t72, np.maximum(0.0, 15.0 - t24), np.maximum(0.0, t24 - 18.0), np.maximum(0.0, 15.0 - temp), np.maximum(0.0, temp - 18.0), rdif, rdir, np.sin(ang), np.cos(ang), np.sin(2 * ang), np.cos(2 * ang), (days - days[0]).astype(np.float64)])

def forecast(item_id, unix, target_past, temp, rdif, rdir, horizon, years=3.0, n_estimators=400):
    import lightgbm as lgb
    L = len(target_past)
    y = np.asarray(target_past, dtype=np.float64)
    positive = np.all(y > 0)
    ly = np.log(y) if positive else y
    X = design(item_id, unix, temp, rdif, rdir)
    if X.shape[0] < L + horizon or not np.isfinite(X).all():
        return None
    ntr = int(min(L, years * 365.25 * 48))
    if ntr < 48 * 60:
        return None
    model = lgb.LGBMRegressor(n_estimators=n_estimators, learning_rate=0.06, num_leaves=63, min_child_samples=40, subsample=0.9, subsample_freq=1, colsample_bytree=0.9, n_jobs=2, verbose=-1)
    model.fit(X[L - ntr:L], ly[L - ntr:L], categorical_feature=CATEGORICAL, feature_name=FEATURES)
    recent = model.predict(X[L - 14 * 48:L])
    bias = 0.5 * np.mean(ly[L - 14 * 48:L] - recent) + 0.5 * np.mean(ly[L - 7 * 48:L] - recent[-7 * 48:])
    out = model.predict(X[L:L + horizon]) + bias
    out = np.exp(out) if positive else out
    return out if np.isfinite(out).all() else None

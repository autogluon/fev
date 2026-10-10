import numpy as np
DAY = 96
WEEK = 7 * DAY
FIXED_HOLIDAYS = {(1, 1), (12, 25), (12, 26), (5, 1), (1, 6), (8, 15), (11, 1), (12, 24), (12, 31)}

def calendar_frame(ts):
    import pandas as pd
    idx = pd.DatetimeIndex(ts)
    return {'slot': (idx.hour * 4 + idx.minute // 15).values.astype(np.int16), 'dow': idx.dayofweek.values.astype(np.int8), 'doy': idx.dayofyear.values.astype(np.int16), 'month': idx.month.values.astype(np.int8), 'day': idx.day.values.astype(np.int8)}

def holiday_flag(cal):
    md = list(zip(cal['month'].tolist(), cal['day'].tolist()))
    return np.array([1.0 if x in FIXED_HOLIDAYS else 0.0 for x in md], dtype=np.float32)

def xmas_window(cal):
    m, d = (cal['month'], cal['day'])
    return ((m == 12) & (d >= 23) | (m == 1) & (d <= 2)).astype(np.float32)

def build_rows(y, cal, hol, xw, known, origins, H=DAY):
    o = np.asarray(origins)[:, None]
    h = np.arange(H)[None, :]
    t = o + h
    feats = {}
    feats['slot'] = cal['slot'][t]
    feats['dow'] = cal['dow'][t]
    feats['doy_sin'] = np.sin(2 * np.pi * cal['doy'][t] / 365.25)
    feats['doy_cos'] = np.cos(2 * np.pi * cal['doy'][t] / 365.25)
    feats['hol'] = hol[t]
    feats['xmas'] = xw[t]
    feats['horizon'] = np.broadcast_to(h, t.shape)
    feats['is_we'] = (cal['dow'][t] >= 5).astype(np.float32)
    lev1 = np.array([y[max(0, int(oo) - DAY):int(oo)].mean() for oo in origins])[:, None]
    lev7 = np.array([y[max(0, int(oo) - WEEK):int(oo)].mean() for oo in origins])[:, None]
    lev28 = np.array([y[max(0, int(oo) - 4 * WEEK):int(oo)].mean() for oo in origins])[:, None]
    base = np.where(np.abs(lev28) < 1e-06, 1.0, lev28)
    lags = {'d1': DAY, 'd2': 2 * DAY, 'd3': 3 * DAY, 'w1': WEEK, 'w2': 2 * WEEK, 'w3': 3 * WEEK, 'w4': 4 * WEEK, 'y1': 52 * WEEK}
    for name, lag in lags.items():
        feats['lag_' + name] = y[np.maximum(t - lag, 0)] / base
    feats['lev1_r'] = np.broadcast_to(lev1 / base, t.shape)
    feats['lev7_r'] = np.broadcast_to(lev7 / base, t.shape)
    wk = np.stack([feats['lag_w1'], feats['lag_w2'], feats['lag_w3'], feats['lag_w4']])
    feats['wk_med'] = np.median(wk, axis=0)
    if known is not None and known.shape[0] >= 3:
        rd, rdir, temp = (known[0], known[1], known[2])
        feats['temp'] = temp[t]
        feats['temp_d1'] = temp[t] - temp[np.maximum(t - DAY, 0)]
        feats['temp_day'] = np.repeat(temp[t].mean(axis=1)[:, None], H, axis=1)
        feats['rad'] = rd[t] + rdir[t]
        feats['rad_day'] = np.repeat((rd[t] + rdir[t]).mean(axis=1)[:, None], H, axis=1)
    names = sorted(feats)
    X = np.stack([np.asarray(feats[n], dtype=np.float32).reshape(-1) for n in names], axis=1)
    return (X, names, base.repeat(H, axis=1).reshape(-1))

def fit_predict(y, ts, known, cutoff, H=DAY, n_origins=520, seed=0, params=None):
    import lightgbm as lgb
    cal = calendar_frame(ts)
    hol = holiday_flag(cal)
    xw = xmas_window(cal)
    lo = 52 * 7 * DAY + DAY
    first = max(lo, cutoff - n_origins * DAY)
    origins = np.arange(first, cutoff - H + 1, DAY)
    if len(origins) < 60:
        origins = np.arange(max(4 * WEEK + DAY, cutoff - n_origins * DAY), cutoff - H + 1, DAY)
    Xtr, names, basetr = build_rows(y, cal, hol, xw, known, origins, H)
    ytr = np.stack([y[o:o + H] for o in origins]).reshape(-1) / basetr
    Xte, _, basete = build_rows(y, cal, hol, xw, known, [cutoff], H)
    p = dict(objective='l1', num_leaves=63, learning_rate=0.06, n_estimators=400, min_child_samples=40, subsample=0.9, subsample_freq=1, colsample_bytree=0.9, reg_lambda=1.0, n_jobs=4, verbose=-1, random_state=seed)
    if params:
        p.update(params)
    age = (cutoff - np.repeat(origins, H)) / float(DAY)
    w = 0.5 ** (age / 365.0)
    m = lgb.LGBMRegressor(**p)
    m.fit(Xtr, ytr, sample_weight=w, categorical_feature=[names.index('dow')])
    pred = m.predict(Xte) * basete
    return (np.asarray(pred, dtype=np.float64), m, names)

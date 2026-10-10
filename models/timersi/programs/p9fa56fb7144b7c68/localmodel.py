import numpy as np
HOUR = 3600
WEEK = 168

def _calendar(ts_int):
    hod = (ts_int % 24).astype(np.float64)
    dow = (ts_int // 24 + 4) % 7
    return (hod, dow.astype(np.float64))

def build_features(logy, know, ts_int, idx, doy, lags=(168, 336, 504, 672, 8736)):
    hod, dow = _calendar(ts_int)
    cols = {}
    h = hod[idx]
    cols['hod'] = h
    cols['dow'] = dow[idx]
    cols['is_we'] = (dow[idx] >= 5).astype(np.float64)
    for n in (1, 2, 3):
        cols[f'hs{n}'] = np.sin(2 * np.pi * n * h / 24.0)
        cols[f'hc{n}'] = np.cos(2 * np.pi * n * h / 24.0)
    d = doy[idx]
    for n in (1, 2, 3, 4, 6, 8):
        cols[f'ys{n}'] = np.sin(2 * np.pi * n * d / 365.25)
        cols[f'yc{n}'] = np.cos(2 * np.pi * n * d / 365.25)
    cols['doy'] = d
    rad_d, rad_b, temp = (know[0], know[1], know[2])
    cols['temp'] = temp[idx]
    cols['cdd'] = np.maximum(temp[idx] - 18.0, 0.0)
    cols['hdd'] = np.maximum(16.0 - temp[idx], 0.0)
    cols['rad_b'] = rad_b[idx]
    cols['rad_d'] = rad_d[idx]
    ct = np.cumsum(np.concatenate([[0.0], temp]))
    for w in (24, 72):
        m = (ct[idx + 1] - ct[np.maximum(idx + 1 - w, 0)]) / np.minimum(idx + 1, w)
        cols[f'temp{w}'] = m
    cols['tanom'] = cols['temp'] - cols['temp72']
    cl = np.cumsum(np.concatenate([[0.0], np.nan_to_num(logy, nan=0.0)]))
    base = (cl[idx - WEEK + 1] - cl[idx - 2 * WEEK + 1]) / WEEK
    for lg in lags:
        v = logy[idx - lg]
        cols[f'l{lg}'] = v - base
    cols['lmean'] = np.mean([cols[f'l{lg}'] for lg in (168, 336, 504, 672)], axis=0)
    base2 = (cl[idx - 2 * WEEK + 1] - cl[idx - 3 * WEEK + 1]) / WEEK
    base4 = (cl[idx - WEEK + 1] - cl[idx - 5 * WEEK + 1]) / (4 * WEEK)
    cols['trend1'] = base - base2
    cols['trend4'] = base - base4
    names = sorted(cols)
    X = np.column_stack([cols[n] for n in names])
    return (X, base, names)

def day_of_year(ts_str):
    t = np.asarray(ts_str).astype('datetime64[h]')
    y0 = t.astype('datetime64[Y]').astype('datetime64[h]')
    return ((t - y0) / np.timedelta64(24, 'h')).astype(np.float64)

def fit_predict(hist, known, timestamps, horizon, n_estimators=400, seed=0, backtest_origins=8, lgb_params=None, seeds=(0,), halflife=None, ridge=False):
    import lightgbm as lgb
    L = hist.shape[0]
    H = horizon
    logy = np.full(L + H, np.nan)
    logy[:L] = np.log(np.maximum(hist, 1e-06))
    t64 = np.asarray(timestamps).astype('datetime64[h]')
    ts_int = (t64 - np.datetime64('1970-01-01T00')) / np.timedelta64(1, 'h')
    ts_int = ts_int.astype(np.int64)
    doy = day_of_year(timestamps)
    minlag = 8736 + WEEK
    params = dict(objective='l2', num_leaves=63, learning_rate=0.05, min_data_in_leaf=40, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=4, n_estimators=n_estimators, seed=seed)
    if lgb_params:
        params.update(lgb_params)

    def sw(tr, end):
        if not halflife:
            return None
        return 0.5 ** ((end - tr) / float(halflife))

    def train_on(series, end):
        tr = np.arange(minlag, end)
        X, base, _ = build_features(series, known, ts_int, tr, doy)
        yv = series[tr] - base
        w = sw(tr, end)
        ms = []
        for sd in seeds:
            pr = dict(params)
            pr['seed'] = sd
            m = lgb.LGBMRegressor(**pr)
            m.fit(X, yv, sample_weight=w)
            ms.append(m)
        if ridge:
            from sklearn.linear_model import Ridge
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler().fit(X)
            rm = Ridge(alpha=10.0).fit(sc.transform(X), yv, sample_weight=w)
            ms.append(('ridge', sc, rm))
        return ms

    def predict_at(models, series, origin):
        te = np.arange(origin, origin + H)
        X, base, _ = build_features(series, known, ts_int, te, doy)
        ps = []
        for m in models:
            if isinstance(m, tuple):
                ps.append(m[2].predict(m[1].transform(X)))
            else:
                ps.append(m.predict(X))
        return np.exp(np.mean(ps, axis=0) + base)
    model = train_on(logy, L)
    point = predict_at(model, logy, L)
    res = []
    for j in range(1, backtest_origins + 1):
        o = L - j * H
        if o - minlag < 2000:
            break
        lg2 = logy.copy()
        lg2[o:] = np.nan
        saved, logy_full = (logy, None)
        try:
            globals()
            mb = train_on(lg2, o)
            pb = predict_at(mb, lg2, o)
            res.append(np.log(np.maximum(hist[o:o + H], 1e-06)) - np.log(np.maximum(pb, 1e-09)))
        finally:
            pass
    res = np.array(res) if res else np.zeros((1, H))
    return {'point': point, 'residuals': res, 'model': model}

def _train_truncated(lg2, known, ts_int, doy, minlag, end, params):
    import lightgbm as lgb
    tr = np.arange(minlag, end)
    X, base, _ = build_features(lg2, known, ts_int, tr, doy)
    yv = lg2[tr] - base
    m = lgb.LGBMRegressor(**params)
    m.fit(X, yv)
    return m

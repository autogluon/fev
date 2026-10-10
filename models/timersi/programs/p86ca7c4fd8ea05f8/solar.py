import numpy as np
import pandas as pd
QL = np.arange(1, 10) / 10.0
KNOWN = ['day_length', 'humidity', 'pressure', 'rain_1h', 'snow_1h', 'temp', 'wind_speed']
PAST = ['clouds_all', 'global_horizontal_irradiance']
PER_HOUR = ['hour', 'lead', 'cs', 'dp', 'cs_rel', 'sin_doy', 'cos_doy'] + KNOWN + ['hum_dev', 'temp_dev', 'press_dev']
PER_ORIGIN = ['w_hum', 'w_temp', 'w_press', 'w_rain', 'w_wind', 'p_idx_today', 'p_idx_d1', 'p_idx_d2', 'p_idx_m7', 'p_clouds_today', 'p_clouds_d1', 'p_idx_trend']
FEATURES = PER_HOUR + PER_ORIGIN

def _circ_table(values, doy, hour, reducer, min_n=6, win=12, smooth=True):
    tab = np.zeros((367, 24))
    for h in range(24):
        m = hour == h
        v = values[m]
        dd = doy[m]
        if v.size == 0:
            continue
        for day in range(1, 367):
            diff = np.abs(dd - day)
            diff = np.minimum(diff, 365 - diff)
            sel = diff <= win
            if int(sel.sum()) >= min_n:
                tab[day, h] = reducer(v[sel])
    if smooth:
        k = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        k /= k.sum()
        body = tab[1:366]
        pad = np.concatenate([body[-2:], body, body[:2]], axis=0)
        sm = np.zeros_like(body)
        for i in range(5):
            sm += k[i] * pad[i:i + body.shape[0]]
        tab[1:366] = sm
    tab[366] = tab[365]
    return tab

def clearsky_table(target, doy, hour, q=0.92, win=12, min_n=6):
    return _circ_table(target, doy, hour, lambda v: np.quantile(v, q), min_n, win, True)

def daylight_prob(target, doy, hour, win=12, min_n=6):
    return _circ_table((target > 0).astype(float), doy, hour, np.mean, min_n, win, False)

def build_features(df, cs_tab, dp_tab, origin_hour=16, H=24):
    idx = df.index
    doy = idx.dayofyear.values
    hour = idx.hour.values
    out = pd.DataFrame(index=idx)
    out['hour'] = hour
    out['cs'] = cs_tab[doy, hour]
    out['dp'] = dp_tab[doy, hour]
    out['sin_doy'] = np.sin(2 * np.pi * doy / 365.25)
    out['cos_doy'] = np.cos(2 * np.pi * doy / 365.25)
    for c in KNOWN:
        out[c] = df[c].values.astype(float) if c in df else np.nan
    target = df['target'].values.astype(float) if 'target' in df else np.full(len(df), np.nan)
    out['target'] = target
    out['index'] = np.where(out['cs'].values > 100, target / np.maximum(out['cs'].values, 1e-06), np.nan)
    origin = (idx - pd.Timedelta(hours=origin_hour + 1)).normalize()
    out['origin'] = origin
    out['lead'] = ((idx - (origin + pd.Timedelta(hours=origin_hour))) / pd.Timedelta(hours=1)).astype(int)
    grp = out.groupby('origin')
    out['cs_rel'] = out['cs'] / grp['cs'].transform('max').clip(lower=1.0)
    for src, dst, how in (('humidity', 'w_hum', 'mean'), ('temp', 'w_temp', 'mean'), ('pressure', 'w_press', 'mean'), ('rain_1h', 'w_rain', 'sum'), ('wind_speed', 'w_wind', 'mean')):
        out[dst] = grp[src].transform(how)
    out['hum_dev'] = out['humidity'] - out['w_hum']
    out['temp_dev'] = out['temp'] - out['w_temp']
    out['press_dev'] = out['pressure'] - out['w_press']
    day = pd.Series(idx.normalize(), index=idx)
    strong = (out['cs'] > 0.25 * out.groupby(day)['cs'].transform('max').clip(lower=1.0)) & (out['cs'] > 100)
    out['_sidx'] = np.where(strong, out['index'], np.nan)
    day_idx = out.groupby(day)['_sidx'].mean()
    today_mask = out['hour'].values <= origin_hour
    tmp = out[['_sidx']].copy()
    tmp['d'] = day.values
    tmp.loc[~today_mask, '_sidx'] = np.nan
    today_idx = tmp.groupby('d')['_sidx'].mean()
    clouds = df['clouds_all'].astype(float) if 'clouds_all' in df else pd.Series(np.nan, index=idx)
    tmpc = pd.DataFrame({'c': clouds.values, 'd': day.values}, index=idx)
    day_clouds = tmpc.groupby('d')['c'].mean()
    tmpc.loc[~today_mask, 'c'] = np.nan
    today_clouds = tmpc.groupby('d')['c'].mean()
    tab = pd.DataFrame({'p_idx_today': today_idx, 'p_clouds_today': today_clouds})
    tab['p_idx_d1'] = day_idx.shift(1)
    tab['p_idx_d2'] = day_idx.shift(2)
    tab['p_idx_m7'] = day_idx.shift(1).rolling(7, min_periods=2).mean()
    tab['p_clouds_d1'] = day_clouds.shift(1)
    tab['p_idx_trend'] = tab['p_idx_today'] - tab['p_idx_m7']
    out = out.join(tab, on='origin')
    return out

def fit_quantiles(X, y, ql=QL, n_estimators=400, seed=0):
    import lightgbm as lgb
    models = []
    for q in ql:
        m = lgb.LGBMRegressor(objective='quantile', alpha=float(q), n_estimators=n_estimators, learning_rate=0.05, num_leaves=31, min_child_samples=40, subsample=0.9, subsample_freq=1, colsample_bytree=0.8, n_jobs=4, verbose=-1, random_state=seed)
        m.fit(X, y)
        models.append(m)
    return models

def predict_quantiles(models, X):
    return np.sort(np.stack([m.predict(X) for m in models], axis=1), axis=1)

def frame_from_view(view):
    ts = pd.DatetimeIndex(np.asarray(view['timestamps']))
    L = int(view['cutoff_index'])
    data = {}
    known_names = list(view['known_names'])
    kf = np.asarray(view['known_features'], dtype=float)
    for j, name in enumerate(known_names):
        data[name] = kf[j]
    past_names = list(view['past_names'])
    pf = np.asarray(view['past_features'], dtype=float)
    for j, name in enumerate(past_names):
        data[name] = np.concatenate([pf[j], np.full(len(ts) - L, np.nan)])
    y = np.asarray(view['target_history'], dtype=float)[0]
    obsv = np.asarray(view['target_observed'])[0].astype(bool)
    y = np.where(obsv, y, np.nan)
    data['target'] = np.concatenate([y, np.full(len(ts) - L, np.nan)])
    df = pd.DataFrame(data, index=ts)
    df = df.reindex(pd.date_range(ts[0], ts[-1], freq='h'))
    for c in KNOWN:
        if c in df:
            df[c] = df[c].interpolate(limit_direction='both')
    for c in PAST:
        if c in df:
            df[c] = df[c].interpolate(limit_direction='both')
    return (df, ts[L:])

def specialist_quantiles(df, future_ts, origin_hour=16, valid_days=365, cs_q=0.92, n_estimators=400, recalibrate=True):
    obs = df['target'].notna().values
    doy = df.index.dayofyear.values
    hour = df.index.hour.values
    cs_tab = clearsky_table(df['target'].values[obs], doy[obs], hour[obs], q=cs_q)
    dp_tab = daylight_prob(df['target'].values[obs], doy[obs], hour[obs])
    F = build_features(df, cs_tab, dp_tab, origin_hour=origin_hour)
    y = F['index'].values
    fin = np.isfinite(y)
    fut = F.index.isin(future_ts)
    cutoff = df.index[obs][-1]
    hist = (F.index <= cutoff) & fin
    delta = np.zeros(9)
    if recalibrate and valid_days:
        vstart = cutoff - pd.Timedelta(days=valid_days)
        trm = hist & (F.index < vstart)
        vam = hist & (F.index >= vstart)
        if trm.sum() > 2000 and vam.sum() > 500:
            M0 = fit_quantiles(F.loc[trm, FEATURES].values, y[trm], n_estimators=n_estimators)
            Qv = predict_quantiles(M0, F.loc[vam, FEATURES].values)
            res = y[vam][:, None] - Qv
            delta = np.array([np.quantile(res[:, i], QL[i]) for i in range(9)])
    M = fit_quantiles(F.loc[hist, FEATURES].values, y[hist], n_estimators=n_estimators)
    Q = predict_quantiles(M, F.loc[fut, FEATURES].values) + delta[None, :]
    Q = np.sort(np.clip(Q, 0.0, None), axis=1)
    cs_f = F.loc[fut, 'cs'].values
    return (Q * cs_f[:, None], F.loc[fut], delta)

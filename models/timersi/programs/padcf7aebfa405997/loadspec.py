import numpy as np
import pandas as pd
try:
    from sklearn.linear_model import Ridge
    _HAVE_SK = True
except Exception:
    _HAVE_SK = False
try:
    import lightgbm as lgb
    _HAVE_LGB = True
except Exception:
    _HAVE_LGB = False

def _candidate_holidays(years):
    out = {}

    def add(rule, d):
        out.setdefault(rule, set()).add(pd.Timestamp(d.date() if hasattr(d, 'date') else d))
    for y in years:
        y = int(y)
        add('newyear', pd.Timestamp(y, 1, 1))
        add('jul4', pd.Timestamp(y, 7, 4))
        add('dec25', pd.Timestamp(y, 12, 25))
        add('dec24', pd.Timestamp(y, 12, 24))
        add('dec31', pd.Timestamp(y, 12, 31))
        add('nov11', pd.Timestamp(y, 11, 11))
        for rule, mo, nth in (('mlk', 1, 3), ('presidents', 2, 3)):
            d = pd.Timestamp(y, mo, 1)
            d = d + pd.Timedelta(days=(7 - d.dayofweek) % 7)
            add(rule, d + pd.Timedelta(days=7 * (nth - 1)))
        d = pd.Timestamp(y, 5, 31)
        add('memorial', d - pd.Timedelta(days=d.dayofweek % 7))
        d = pd.Timestamp(y, 9, 1)
        add('labor', d + pd.Timedelta(days=(7 - d.dayofweek) % 7))
        d = pd.Timestamp(y, 11, 1)
        d = d + pd.Timedelta(days=(3 - d.dayofweek) % 7) + pd.Timedelta(days=21)
        add('thanksgiving', d)
        add('blackfriday', d + pd.Timedelta(days=1))
    return out
MAJOR_HOLIDAY_PRIOR = frozenset(('newyear', 'memorial', 'jul4', 'thanksgiving', 'dec24', 'dec25'))
HOL_RATIO_MAX = 0.97
_HOL_VALID = None

def measure_holiday_rules(timestamps_past, y_past):
    t = pd.DatetimeIndex(pd.to_datetime(pd.Series(np.asarray(timestamps_past))).values)
    y = np.asarray(y_past, float)
    n = min(len(t), len(y))
    df = pd.DataFrame({'y': y[:n]}, index=t[:n])
    df = df[np.isfinite(df['y'])]
    daily = df['y'].resample('D').agg(['mean', 'count'])
    daily = daily[daily['count'] >= 20]
    if daily.empty:
        return (set(MAJOR_HOLIDAY_PRIOR), {'measured': {}})
    cand = _candidate_holidays(sorted(set(t.year)))
    all_cand = set().union(*cand.values())
    dow = daily.index.dayofweek
    valid, diag = (set(), {})
    for rule, dates in cand.items():
        ratios = []
        for d in sorted(dates):
            if d not in daily.index:
                continue
            same = daily[(dow == d.dayofweek) & (abs((daily.index - d).days) <= 21) & ~daily.index.isin(all_cand)]
            if len(same) < 2:
                continue
            ratios.append(float(daily.loc[d, 'mean'] / max(same['mean'].median(), 1e-09)))
        if ratios:
            r = float(np.median(ratios))
            diag[rule] = round(r, 4)
            if r <= HOL_RATIO_MAX:
                valid.add(rule)
        elif rule in MAJOR_HOLIDAY_PRIOR:
            valid.add(rule)
            diag[rule] = 'prior'
    return (valid, {'measured': diag})

def set_measured_holidays(timestamps_past, y_past):
    global _HOL_VALID
    try:
        _HOL_VALID, diag = measure_holiday_rules(timestamps_past, y_past)
    except Exception:
        _HOL_VALID, diag = (None, {'measured': 'error'})
    return (_HOL_VALID, diag)

def arm_measured_holidays(timestamps, L, y_past):
    global _HOL_VALID
    _HOL_VALID = None
    try:
        t = pd.DatetimeIndex(pd.to_datetime(pd.Series(np.asarray(timestamps))).values)
        fut = set(t[L:].normalize())
        cand = _candidate_holidays(sorted(set(t.year)))
        hit = sorted({r for r, ds in cand.items() if ds & fut})
        if not hit:
            return (None, {'armed': False, 'horizon_holidays': []})
        valid, diag = set_measured_holidays(t[:L], y_past)
        diag = dict(diag, armed=True, horizon_holidays=hit)
        return (valid, diag)
    except Exception:
        _HOL_VALID = None
        return (None, {'armed': False, 'error': True})

def _rule_holidays(years):
    cand = _candidate_holidays(years)
    if _HOL_VALID is None:
        return set().union(*cand.values())
    out = set()
    for rule in _HOL_VALID:
        out |= cand.get(rule, set())
    return out
_CLOCK_HOUR = False

def _dst_flag(t):
    flag = np.zeros(len(t), bool)
    for y in sorted(set(t.year)):
        d = pd.Timestamp(int(y), 3, 1)
        start = d + pd.Timedelta(days=(6 - d.dayofweek) % 7 + 7, hours=2)
        d = pd.Timestamp(int(y), 11, 1)
        end = d + pd.Timedelta(days=(6 - d.dayofweek) % 7, hours=2)
        flag |= (t >= start) & (t < end)
    return flag

def calendar_frame(timestamps):
    t = pd.DatetimeIndex(pd.to_datetime(pd.Series(np.asarray(timestamps))).values)
    hol = _rule_holidays(sorted(set(t.year)))
    hour = t.hour.to_numpy()
    if _CLOCK_HOUR:
        hour = (hour + _dst_flag(t).astype(int)) % 24
    return pd.DataFrame({'hour': hour, 'dow': t.dayofweek, 'doy': t.dayofyear, 'hol': pd.Series(t.normalize()).isin(hol).to_numpy(float)})

def temp_block(temp):
    s = pd.Series(np.asarray(temp, float))
    out = {'t': s.to_numpy(), 't_ma24': s.rolling(24, min_periods=1).mean().to_numpy(), 't_ma168': s.rolling(168, min_periods=1).mean().to_numpy(), 't_ma48': s.rolling(48, min_periods=1).mean().to_numpy(), 't_max24': s.rolling(24, min_periods=1).max().to_numpy(), 't_min24': s.rolling(24, min_periods=1).min().to_numpy()}
    for lag in LAGS_T:
        out['t_l%d' % lag] = s.shift(lag).bfill().to_numpy()
    return out
LAGS_T = (1, 2, 3, 6, 12, 24, 48)
CORR_DECAY = 1.0

def _decay_vec(horizon):
    return CORR_DECAY ** (np.arange(horizon) // 24)
RECENCY_KEYS = ('t_l1', 't_l2', 't_l3', 't_l6', 't_l12', 't_l24', 't_l48', 't_max24', 't_min24', 't_ma48')

def safe_lags(y_past, horizon, lags=(168, 336)):
    L = len(y_past)
    fill = float(np.nanmean(y_past))
    full = np.concatenate([y_past, np.full(horizon, np.nan)])
    out = []
    for lag in lags:
        if lag < horizon:
            continue
        v = np.concatenate([np.full(lag, y_past[0]), full[:-lag]])
        v = np.where(np.isfinite(v), v, fill)
        out.append(np.log(np.maximum(v, 1e-06)))
    return out

def design_matrix(df, tb, lag_cols):
    n = len(df)
    cols = []

    def add(v):
        cols.append(np.asarray(v, float).reshape(n, -1))
    hour = df['hour'].to_numpy(int)
    dow = df['dow'].to_numpy(int)
    doy = df['doy'].to_numpy(float)
    for h in (1, 2, 3):
        add(np.sin(2 * np.pi * h * doy / 365.25))
        add(np.cos(2 * np.pi * h * doy / 365.25))
    daytype = np.where(df['hol'].to_numpy() > 0, 2, np.where(dow >= 5, 1, 0))
    hd = np.zeros((n, 72))
    hd[np.arange(n), hour * 3 + daytype] = 1.0
    add(hd)
    dw = np.zeros((n, 7))
    dw[np.arange(n), dow] = 1.0
    add(dw)
    t = (tb['t'] - 55.0) / 20.0
    tm = (tb['t_ma24'] - 55.0) / 20.0
    tw = (tb['t_ma168'] - 55.0) / 20.0
    for p in (1, 2, 3):
        add(t ** p)
        add(tm ** p)
    add(tw)
    hh = np.zeros((n, 24))
    hh[np.arange(n), hour] = 1.0
    for base in (t, t ** 2, t ** 3):
        add(hh * base[:, None])
    for base in (tm, tm ** 2):
        add(hh * base[:, None])
    for base in (t, t ** 2):
        add(base * np.sin(2 * np.pi * doy / 365.25))
        add(base * np.cos(2 * np.pi * doy / 365.25))
    for key in RECENCY_KEYS:
        if key in tb:
            v = (tb[key] - 55.0) / 20.0
            add(v)
            add(v ** 2)
    add(np.arange(n, dtype=float) / max(n, 1))
    for c in lag_cols:
        add(c)
    return np.hstack(cols)

def _recent_correction(resid_log, hour_past, wknd_past, hour_fut, wknd_fut, mask):
    g = float(np.mean(resid_log[mask])) if mask.any() else 0.0
    table = np.full((24, 2), g)
    for h in range(24):
        for w in (0, 1):
            sel = mask & (hour_past == h) & (wknd_past == bool(w))
            if sel.sum() >= 3:
                table[h, w] = float(np.mean(resid_log[sel]))
    return (table[hour_fut, wknd_fut.astype(int)], g)

def ridge_forecast(timestamps, y_past, temp_all, horizon, alpha=10.0, recent_days=28, shrink=0.75):
    L = len(y_past)
    df = calendar_frame(timestamps)
    tb = temp_block(temp_all)
    X = design_matrix(df, tb, safe_lags(y_past, horizon))
    Xtr, Xte = (X[:L], X[L:L + horizon])
    ylog = np.log(np.maximum(y_past, 1e-06))
    msk = np.isfinite(ylog) & np.isfinite(Xtr).all(1)
    mu = Xtr[msk].mean(0)
    sd = Xtr[msk].std(0)
    sd[sd < 1e-09] = 1.0
    m = Ridge(alpha=alpha)
    m.fit((Xtr[msk] - mu) / sd, ylog[msk])
    pl = m.predict((Xte - mu) / sd)
    fl = m.predict((Xtr - mu) / sd)
    n = int(min(recent_days * 24, L))
    r = (ylog - fl)[-n:]
    hp = df['hour'].to_numpy(int)[L - n:L]
    wp = df['dow'].to_numpy(int)[L - n:L] >= 5
    hf = df['hour'].to_numpy(int)[L:L + horizon]
    wf = df['dow'].to_numpy(int)[L:L + horizon] >= 5
    adj, g = _recent_correction(r, hp, wp, hf, wf, msk[-n:] & np.isfinite(r))
    return (np.exp(pl + shrink * adj), g)

def hourwise_fit(timestamps, y_past, temp_all, horizon, alpha=3.0, recent_days=28, shrink=0.75):
    L = len(y_past)
    df = calendar_frame(timestamps)
    tb = temp_block(temp_all)
    n = len(df)
    hour = df['hour'].to_numpy(int)
    dow = df['dow'].to_numpy(int)
    doy = df['doy'].to_numpy(float)
    hol = df['hol'].to_numpy(float)
    sc = lambda v: (v - 55.0) / 20.0
    t, tm, tw = (sc(tb['t']), sc(tb['t_ma24']), sc(tb['t_ma168']))
    t48, tmx, tmn = (sc(tb['t_ma48']), sc(tb['t_max24']), sc(tb['t_min24']))
    t24, t3 = (sc(tb['t_l24']), sc(tb['t_l3']))
    dw = np.zeros((n, 7))
    dw[np.arange(n), dow] = 1.0
    cols = [dw, hol[:, None]]
    for h in (1, 2, 3):
        cols += [np.sin(2 * np.pi * h * doy / 365.25)[:, None], np.cos(2 * np.pi * h * doy / 365.25)[:, None]]
    for v in (t, tm, tw, t48, tmx, tmn, t24, t3):
        cols += [v[:, None], (v ** 2)[:, None], (v ** 3)[:, None]]
    sa, ca = (np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25))
    for v in (t, tm):
        cols += [(v * sa)[:, None], (v * ca)[:, None], (v ** 2 * sa)[:, None], (v ** 2 * ca)[:, None]]
    cols += [(np.arange(n, dtype=float) / max(n, 1))[:, None]]
    X = np.hstack(cols)
    ylog = np.log(np.maximum(y_past, 1e-06))
    base = np.zeros(horizon)
    fit = np.full(L, np.nan)
    for h in range(24):
        itr = np.where(hour[:L] == h)[0]
        ite = np.where(hour[L:L + horizon] == h)[0] + L
        if len(itr) < 40 or len(ite) == 0:
            continue
        Xt = X[itr]
        mu = Xt.mean(0)
        sd = Xt.std(0)
        sd[sd < 1e-09] = 1.0
        m = Ridge(alpha=alpha)
        m.fit((Xt - mu) / sd, ylog[itr])
        base[ite - L] = m.predict((X[ite] - mu) / sd)
        fit[itr] = m.predict((Xt - mu) / sd)
    nn = int(min(recent_days * 24, L))
    r = (ylog - fit)[-nn:]
    adj, _ = _recent_correction(r, hour[L - nn:L], dow[L - nn:L] >= 5, hour[L:L + horizon], dow[L:L + horizon] >= 5, np.isfinite(r))
    return (np.exp(base + shrink * adj), fit, base + shrink * adj)

def gbm_residual_forecast(timestamps, y_past, temp_all, horizon, fit_log, fore_log, n_estimators=300, learning_rate=0.05, num_leaves=31):
    L = len(y_past)
    df = calendar_frame(timestamps)
    tb = temp_block(temp_all)
    doy = df['doy'].to_numpy(float)
    lags = safe_lags(y_past, horizon)
    feats = [df['hour'].to_numpy(float), df['dow'].to_numpy(float), df['hol'].to_numpy(float), tb['t'], tb['t_ma24'], tb['t_ma168'], tb['t'] - tb['t_ma24'], np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25), np.arange(len(df), dtype=float) / max(len(df), 1)] + list(lags) + [tb[k] for k in ('t_l1', 't_l3', 't_l6', 't_l12', 't_l24', 't_max24', 't_min24', 't_ma48') if k in tb]
    X = np.column_stack(feats)
    res = np.log(np.maximum(y_past, 1e-06)) - fit_log
    ok = np.isfinite(res)
    model = lgb.LGBMRegressor(n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leaves, min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=4, random_state=0)
    model.fit(X[:L][ok], res[ok], categorical_feature=[0, 1])
    return np.exp(fore_log + model.predict(X[L:L + horizon]))

def balance_point(timestamps, y_past, temp_past, lo=48.0, hi=72.0):
    L = min(len(y_past), len(temp_past))
    nd = L // 24
    if nd < 60:
        return 65.0
    yd = np.nanmean(np.asarray(y_past[:nd * 24], float).reshape(nd, 24), axis=1)
    td = np.nanmean(np.asarray(temp_past[:nd * 24], float).reshape(nd, 24), axis=1)
    ok = np.isfinite(yd) & np.isfinite(td)
    yd, td = (yd[ok], td[ok])
    if yd.size < 60:
        return 65.0
    best, best_sse = (65.0, None)
    for b in np.arange(lo, hi + 0.1, 1.0):
        A = np.column_stack([np.ones_like(td), np.maximum(td - b, 0.0), np.maximum(b - td, 0.0)])
        coef, *_ = np.linalg.lstsq(A, yd, rcond=None)
        sse = float(np.sum((yd - A @ coef) ** 2))
        if best_sse is None or sse < best_sse:
            best, best_sse = (float(b), sse)
    return best

def gbm_forecast(timestamps, y_past, temp_all, horizon, n_estimators=300, learning_rate=0.06, num_leaves=31):
    L = len(y_past)
    df = calendar_frame(timestamps)
    tb = temp_block(temp_all)
    lags = safe_lags(y_past, horizon)
    doy = df['doy'].to_numpy(float)
    feats = [df['hour'].to_numpy(float), df['dow'].to_numpy(float), df['hol'].to_numpy(float), tb['t'], tb['t_ma24'], tb['t_ma168'], tb['t'] - tb['t_ma24'], np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25), np.arange(len(df), dtype=float) / max(len(df), 1)] + list(lags) + [tb[k] for k in ('t_l1', 't_l3', 't_l6', 't_l12', 't_l24', 't_max24', 't_min24', 't_ma48') if k in tb]
    X = np.column_stack(feats)
    ylog = np.log(np.maximum(y_past, 1e-06))
    msk = np.isfinite(ylog)
    model = lgb.LGBMRegressor(n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leaves, min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=4, random_state=0)
    model.fit(X[:L][msk], ylog[msk], categorical_feature=[0, 1])
    return np.exp(model.predict(X[L:L + horizon]))

def hourwise_forecast(timestamps, y_past, temp_all, horizon, alpha=3.0, recent_days=28, shrink=0.75):
    L = len(y_past)
    df = calendar_frame(timestamps)
    tb = temp_block(temp_all)
    n = len(df)
    hour = df['hour'].to_numpy(int)
    dow = df['dow'].to_numpy(int)
    doy = df['doy'].to_numpy(float)
    hol = df['hol'].to_numpy(float)
    sc = lambda v: (v - 55.0) / 20.0
    t, tm, tw = (sc(tb['t']), sc(tb['t_ma24']), sc(tb['t_ma168']))
    t48, tmx, tmn = (sc(tb['t_ma48']), sc(tb['t_max24']), sc(tb['t_min24']))
    t24, t3 = (sc(tb['t_l24']), sc(tb['t_l3']))
    dw = np.zeros((n, 7))
    dw[np.arange(n), dow] = 1.0
    cols = [dw, hol[:, None]]
    for h in (1, 2, 3):
        cols += [np.sin(2 * np.pi * h * doy / 365.25)[:, None], np.cos(2 * np.pi * h * doy / 365.25)[:, None]]
    for v in (t, tm, tw, t48, tmx, tmn, t24, t3):
        cols += [v[:, None], (v ** 2)[:, None], (v ** 3)[:, None]]
    sa, ca = (np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25))
    for v in (t, tm):
        cols += [(v * sa)[:, None], (v * ca)[:, None], (v ** 2 * sa)[:, None], (v ** 2 * ca)[:, None]]
    cols += [(np.arange(n, dtype=float) / max(n, 1))[:, None]]
    X = np.hstack(cols)
    ylog = np.log(np.maximum(y_past, 1e-06))
    out = np.zeros(horizon)
    fit = np.full(L, np.nan)
    for h in range(24):
        itr = np.where(hour[:L] == h)[0]
        ite = np.where(hour[L:L + horizon] == h)[0] + L
        if len(itr) < 40 or len(ite) == 0:
            continue
        Xt = X[itr]
        mu = Xt.mean(0)
        sd = Xt.std(0)
        sd[sd < 1e-09] = 1.0
        m = Ridge(alpha=alpha)
        m.fit((Xt - mu) / sd, ylog[itr])
        out[ite - L] = m.predict((X[ite] - mu) / sd)
        fit[itr] = m.predict((Xt - mu) / sd)
    nn = int(min(recent_days * 24, L))
    r = (ylog - fit)[-nn:]
    adj, _ = _recent_correction(r, hour[L - nn:L], dow[L - nn:L] >= 5, hour[L:L + horizon], dow[L:L + horizon] >= 5, np.isfinite(r))
    return np.exp(out + shrink * adj)

def specialist_forecast(timestamps, y_past, temp_all, horizon):
    y_past = np.asarray(y_past, float)
    L = len(y_past)
    info = {'parts': []}
    if horizon <= 0 or L < 24 * 120:
        return (None, {'reason': 'short_history', 'L': L})
    finite = np.isfinite(y_past)
    if finite.sum() < 0.8 * L or np.nanmin(y_past[finite]) <= 0:
        return (None, {'reason': 'unusable_target'})
    y_filled = y_past.copy()
    if not finite.all():
        y_filled = pd.Series(y_filled).ffill().bfill().to_numpy()
    temp_all = np.asarray(temp_all, float)
    if temp_all.shape[0] < L + horizon or not np.all(np.isfinite(temp_all[:L + horizon])):
        return (None, {'reason': 'temperature_unavailable'})
    temp_all = temp_all[:L + horizon]
    preds = []
    if _HAVE_SK:
        try:
            p, g = ridge_forecast(timestamps, y_filled, temp_all, horizon)
            if np.all(np.isfinite(p)):
                preds.append(p)
                info['parts'].append('ridge')
                info['recent_bias_log'] = float(g)
        except Exception as exc:
            info['ridge_error'] = type(exc).__name__
    hw_fit = None
    if _HAVE_SK:
        try:
            p, fit_log, fore_log = hourwise_fit(timestamps, y_filled, temp_all, horizon)
            if np.all(np.isfinite(p)):
                preds.append(p)
                info['parts'].append('hourwise')
                hw_fit = (fit_log, fore_log)
        except Exception as exc:
            info['hourwise_error'] = type(exc).__name__
    if _HAVE_LGB and hw_fit is not None:
        try:
            p = gbm_residual_forecast(timestamps, y_filled, temp_all, horizon, *hw_fit)
            if np.all(np.isfinite(p)):
                preds.append(p)
                info['parts'].append('gbm_residual')
        except Exception as exc:
            info['gbm_residual_error'] = type(exc).__name__
    if _HAVE_LGB:
        try:
            p = gbm_forecast(timestamps, y_filled, temp_all, horizon)
            if np.all(np.isfinite(p)):
                preds.append(p)
                info['parts'].append('gbm')
        except Exception as exc:
            info['gbm_error'] = type(exc).__name__
    if not preds:
        return (None, dict(info, reason='all_specialists_failed'))
    out = np.mean(np.vstack(preds), axis=0)
    info['reason'] = 'ok'
    return (out, info)

def rolling_oos_load(timestamps, y_past, temp_all, block=672, min_train=24 * 180):
    L = len(y_past)
    out = np.full(L, np.nan)
    start = int(min(min_train, max(24 * 150, L // 3)))
    if start >= L:
        return None
    first = True
    for s in range(start, L, block):
        blk = int(min(block, L - s))
        p, fit_log, _ = hourwise_fit(timestamps[:s + blk], y_past[:s], temp_all[:s + blk], blk)
        out[s:s + blk] = p
        if first:
            warm = np.exp(fit_log[:s])
            out[:s] = np.where(np.isfinite(warm), warm, np.nanmean(y_past[:s]))
            first = False
    if not np.all(np.isfinite(out)):
        out = np.where(np.isfinite(out), out, float(np.nanmean(y_past)))
    return out

def short_forecast(timestamps, y_past, temp_all, horizon, alpha=5.0):
    y_past = np.asarray(y_past, float)
    L = len(y_past)
    if horizon <= 0 or L < 14 * 24 or (not _HAVE_SK):
        return (None, {'reason': 'short_history_too_short', 'L': L})
    finite = np.isfinite(y_past)
    if finite.sum() < 0.8 * L or np.nanmin(y_past[finite]) <= 0:
        return (None, {'reason': 'unusable_target'})
    y = pd.Series(y_past).ffill().bfill().to_numpy()
    temp_all = np.asarray(temp_all, float)
    if temp_all.shape[0] < L + horizon or not np.all(np.isfinite(temp_all[:L + horizon])):
        return (None, {'reason': 'temperature_unavailable'})
    n = L + horizon
    df = calendar_frame(timestamps[:n])
    tb = temp_block(temp_all[:n])
    hour = df['hour'].to_numpy(int)
    wk = (df['dow'].to_numpy(int) >= 5).astype(float)
    hol = df['hol'].to_numpy(float)
    hh = np.zeros((n, 24))
    hh[np.arange(n), hour] = 1.0
    sc = lambda v: (v - 55.0) / 20.0
    t, tm = (sc(tb['t']), sc(tb['t_ma24']))
    t24, t3 = (sc(tb['t_l24']), sc(tb['t_l3']))
    cols = [hh, hh * wk[:, None], hol[:, None], wk[:, None]]
    for v in (t, tm, t24, t3):
        cols += [v[:, None], (v ** 2)[:, None]]
    cols += [hh * t[:, None]]
    X = np.hstack(cols)
    ylog = np.log(np.maximum(y, 1e-06))
    ok = np.isfinite(ylog[:L]) & np.isfinite(X[:L]).all(1)
    if ok.sum() < 200:
        return (None, {'reason': 'too_few_rows'})
    mu = X[:L][ok].mean(0)
    sd = X[:L][ok].std(0)
    sd[sd < 1e-09] = 1.0
    m = Ridge(alpha=alpha)
    m.fit((X[:L][ok] - mu) / sd, ylog[:L][ok])
    p = np.exp(m.predict((X[L:L + horizon] - mu) / sd))
    if not np.all(np.isfinite(p)):
        return (None, {'reason': 'nonfinite_prediction'})
    return (p, {'reason': 'ok', 'parts': ['short_ridge'], 'rows': int(ok.sum())})

def any_specialist(timestamps, y_past, temp_all, horizon):
    L = len(y_past)
    arm_measured_holidays(timestamps, L, y_past)
    if L >= 120 * 24:
        p, info = specialist_forecast(timestamps, y_past, temp_all, horizon)
        return (p, info, 'full')
    if L >= 336:
        p, info = short_forecast(timestamps, y_past, temp_all, horizon)
        return (p, info, 'short')
    return (None, {'reason': 'short_history', 'L': L}, 'none')

def simtemp_sel(temp_all, L, horizon, n_days=28, look=90):
    t = np.asarray(temp_all, float)
    nd = L // 24
    base = L - nd * 24
    if nd < 10:
        sel = np.zeros(L, bool)
        sel[-min(n_days * 24, L):] = True
        return sel
    dmt = np.nanmean(t[base:L].reshape(nd, 24), 1)
    fmt = float(np.nanmean(t[L:L + horizon]))
    order = np.argsort(np.abs(dmt - fmt))
    cand = [d for d in order if d >= nd - look][:n_days]
    sel = np.zeros(L, bool)
    for d in sorted(cand):
        sel[base + d * 24:base + (d + 1) * 24] = True
    return sel

def _apply_corr(resid_log, df, L, horizon, sel, shrink=0.75):
    hour = df['hour'].to_numpy(int)
    dow = df['dow'].to_numpy(int)
    r = resid_log[sel]
    adj, _ = _recent_correction(r, hour[:L][sel], dow[:L][sel] >= 5, hour[L:L + horizon], dow[L:L + horizon] >= 5, np.isfinite(r))
    return adj

def specialist_variants(timestamps, y_past, temp_all, horizon):
    y_past = np.asarray(y_past, float)
    L = len(y_past)
    info = {'parts': []}
    if horizon <= 0 or L < 24 * 120:
        return (None, {'reason': 'short_history', 'L': L})
    finite = np.isfinite(y_past)
    if finite.sum() < 0.8 * L or np.nanmin(y_past[finite]) <= 0:
        return (None, {'reason': 'unusable_target'})
    y = pd.Series(y_past).ffill().bfill().to_numpy()
    temp_all = np.asarray(temp_all, float)
    if temp_all.shape[0] < L + horizon or not np.all(np.isfinite(temp_all[:L + horizon])):
        return (None, {'reason': 'temperature_unavailable'})
    temp_all = temp_all[:L + horizon]
    df = calendar_frame(timestamps[:L + horizon])
    ylog = np.log(np.maximum(y, 1e-06))
    nrec = int(min(28 * 24, L))
    sel_rec = np.zeros(L, bool)
    sel_rec[-nrec:] = True
    sel_sim = simtemp_sel(temp_all, L, horizon)
    preds = {'recent': [], 'simtemp': []}
    if _HAVE_SK:
        try:
            tb = temp_block(temp_all)
            X = design_matrix(df, tb, safe_lags(y, horizon))
            msk = np.isfinite(ylog) & np.isfinite(X[:L]).all(1)
            mu = X[:L][msk].mean(0)
            sd = X[:L][msk].std(0)
            sd[sd < 1e-09] = 1.0
            m = Ridge(alpha=10.0)
            m.fit((X[:L][msk] - mu) / sd, ylog[msk])
            pl = m.predict((X[L:L + horizon] - mu) / sd)
            fl = m.predict((X[:L] - mu) / sd)
            resid = ylog - fl
            for nm, sel in (('recent', sel_rec), ('simtemp', sel_sim)):
                preds[nm].append(np.exp(pl + 0.75 * _apply_corr(resid, df, L, horizon, sel) * _decay_vec(horizon)))
            info['parts'].append('ridge')
        except Exception as exc:
            info['ridge_error'] = type(exc).__name__
        hw = None
        try:
            ph, fit_log, fore_log = hourwise_fit(timestamps, y, temp_all, horizon)
            resid = ylog - fit_log
            adj_rec = _apply_corr(resid, df, L, horizon, sel_rec)
            base_log = fore_log - 0.75 * adj_rec
            adj_sim = _apply_corr(resid, df, L, horizon, sel_sim)
            dv = _decay_vec(horizon)
            hw = {'recent': base_log + 0.75 * adj_rec * dv, 'simtemp': base_log + 0.75 * adj_sim * dv}
            for nm in preds:
                preds[nm].append(np.exp(hw[nm]))
            info['parts'].append('hourwise')
            if _HAVE_LGB:
                try:
                    import lightgbm as _lgb
                    tb2 = temp_block(temp_all)
                    doy = df['doy'].to_numpy(float)
                    lags = safe_lags(y, horizon)
                    feats = [df['hour'].to_numpy(float), df['dow'].to_numpy(float), df['hol'].to_numpy(float), tb2['t'], tb2['t_ma24'], tb2['t_ma168'], tb2['t'] - tb2['t_ma24'], np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25), np.arange(len(df), dtype=float) / max(len(df), 1)] + list(lags) + [tb2[k] for k in ('t_l1', 't_l3', 't_l6', 't_l12', 't_l24', 't_max24', 't_min24', 't_ma48')]
                    Xg = np.column_stack(feats)
                    res = ylog[:L] - fit_log
                    ok = np.isfinite(res)
                    gm = _lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=4, random_state=0)
                    gm.fit(Xg[:L][ok], res[ok], categorical_feature=[0, 1])
                    gfut = gm.predict(Xg[L:L + horizon])
                    for nm in preds:
                        preds[nm].append(np.exp(hw[nm] + gfut))
                    info['parts'].append('gbm_residual')
                except Exception as exc:
                    info['gbm_residual_error'] = type(exc).__name__
        except Exception as exc:
            info['hourwise_error'] = type(exc).__name__
    if _HAVE_LGB:
        try:
            pg = gbm_forecast(timestamps, y, temp_all, horizon)
            for nm in preds:
                preds[nm].append(pg)
            info['parts'].append('gbm')
        except Exception as exc:
            info['gbm_error'] = type(exc).__name__
    if not preds['recent']:
        return (None, dict(info, reason='all_specialists_failed'))
    ext_hours = _extreme_day_mask(temp_all, L, horizon)
    keep = [i for i, nm in enumerate(info['parts']) if nm in GUARD_EXPERTS]
    info['extreme_days'] = int(ext_hours.sum() // 24)
    info['guard_members'] = [info['parts'][i] for i in keep]
    out = {}
    for nm, v in preds.items():
        if not v:
            continue
        arr = np.vstack(v)
        if not np.all(np.isfinite(arr)):
            continue
        comb = arr.mean(axis=0)
        if ext_hours.any() and len(keep) >= 1 and (len(keep) < arr.shape[0]):
            comb = np.where(ext_hours, arr[keep].mean(axis=0), comb)
        out[nm] = comb
    if not out:
        return (None, dict(info, reason='nonfinite'))
    info['reason'] = 'ok'
    return (out, info)
GUARD_EXPERTS = ('ridge', 'gbm_residual')
EXT_HI_PCT, EXT_LO_PCT = (95.0, 5.0)

def _extreme_day_mask(temp_all, L, horizon):
    t = np.asarray(temp_all, float)
    nd = L // 24
    nfd = max(1, horizon // 24)
    mask = np.zeros(horizon, bool)
    if nd < 60 or len(t) < L + horizon:
        return mask
    base = L - nd * 24
    dmax = np.nanmax(t[base:L].reshape(nd, 24), 1)
    dmin = np.nanmin(t[base:L].reshape(nd, 24), 1)
    hi = np.nanpercentile(dmax, EXT_HI_PCT)
    lo = np.nanpercentile(dmin, EXT_LO_PCT)
    for d in range(nfd):
        a = L + d * 24
        b = min(L + horizon, a + 24)
        fmax, fmin = (np.nanmax(t[a:b]), np.nanmin(t[a:b]))
        if fmax >= hi or fmin <= lo:
            mask[d * 24:d * 24 + (b - a)] = True
    return mask

def any_specialist_variants(timestamps, y_past, temp_all, horizon):
    L = len(y_past)
    arm_measured_holidays(timestamps, L, y_past)
    if L >= 120 * 24:
        out, info = specialist_variants(timestamps, y_past, temp_all, horizon)
        return (out, info, 'full')
    if L >= 336:
        p, info = short_forecast(timestamps, y_past, temp_all, horizon)
        return (None if p is None else {'recent': p}, info, 'short')
    return (None, {'reason': 'short_history', 'L': L}, 'none')

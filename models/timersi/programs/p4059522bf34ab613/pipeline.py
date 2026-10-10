import numpy as np
try:
    import pandas as pd
except Exception:
    pd = None
DOW_WEIGHT = 0.5
DOW_LOOKBACK = 364
DOW_HALFLIFE = 120.0
MIN_DOW_OBS = 4
LOWER_SHRINK = 0.6
MAX_LOG_ADJ = 0.2

def _dates(view):
    if pd is None:
        return None
    return pd.to_datetime(np.asarray(view['timestamps']))

def _weekly_profile(values, dow, weights=None, win=7):
    v = np.asarray(values, dtype=float)
    ok = np.isfinite(v) & (v > 0)
    if ok.sum() < 2 * win:
        return (None, None)
    lv = np.full(v.shape, np.nan)
    lv[ok] = np.log(v[ok])
    idx = np.arange(len(lv))
    filled = np.interp(idx, idx[ok], lv[ok])
    half = win // 2
    pad = np.concatenate([np.full(half, filled[0]), filled, np.full(half, filled[-1])])
    kern = np.ones(win) / win
    base = np.convolve(pad, kern, mode='valid')
    resid = lv - base
    prof = np.zeros(7)
    cnt = np.zeros(7)
    wsum = np.zeros(7)
    w = np.ones(len(lv)) if weights is None else np.asarray(weights, dtype=float)
    for d in range(7):
        sel = (dow == d) & np.isfinite(resid)
        cnt[d] = sel.sum()
        if cnt[d] > 0:
            wsum[d] = w[sel].sum()
            prof[d] = float(np.sum(resid[sel] * w[sel]) / max(wsum[d], 1e-09))
    return (prof, cnt)

def _dow_adjustment(hist_vals, hist_dow, hist_age, fc_vals, fc_dow):
    w = np.exp(-np.log(2.0) * hist_age / DOW_HALFLIFE)
    hp, hc = _weekly_profile(hist_vals, hist_dow, w)
    fp, fc = _weekly_profile(fc_vals, fc_dow)
    if hp is None or fp is None:
        return None
    usable = (hc >= MIN_DOW_OBS) & (fc >= 2)
    if usable.sum() < 4:
        return None
    hp = hp - hp[usable].mean()
    fp = fp - fp[usable].mean()
    delta = np.where(usable, hp - fp, 0.0)
    adj = DOW_WEIGHT * delta[fc_dow]
    adj = adj - adj.mean()
    return np.clip(adj, -MAX_LOG_ADJ, MAX_LOG_ADJ)
RAGGED_WIN = 28
RAGGED_THR = 0.6
RAGGED_MINLEN = 180
TRIM_LAUNCH = False
YOY_LAG = 364
YOY_MIN_OVERLAP = 240

def _log_bridge(values, observed):
    v = np.asarray(values, dtype=float).copy()
    ok = np.asarray(observed, dtype=bool) & np.isfinite(v) & (v > 0)
    if ok.all() or ok.sum() < 14:
        return v
    idx = np.arange(len(v))
    v[~ok] = np.exp(np.interp(idx[~ok], idx[ok], np.log(v[ok])))
    return v

def _dense_start(observed):
    o = np.asarray(observed, dtype=float)
    L = len(o)
    if L < RAGGED_WIN + RAGGED_MINLEN:
        return 0
    c = np.concatenate([[0.0], np.cumsum(o)])
    frac = (c[RAGGED_WIN:] - c[:-RAGGED_WIN]) / RAGGED_WIN
    bad = np.where(frac < RAGGED_THR)[0]
    if len(bad) == 0:
        return 0
    s = int(bad[-1]) + RAGGED_WIN
    return s if L - s >= RAGGED_MINLEN else 0

def _quantile_shrink(n_levels=9, lower=LOWER_SHRINK):
    a = np.ones(n_levels)
    mid = n_levels // 2
    a[:mid] = lower
    return a

def preprocess(view, card):
    state = {'N': len(view['target_ids']), 'H': view['horizon'], 'L': view['cutoff_index'], 'card': card}
    dates = _dates(view)
    if dates is not None:
        state['dow'] = dates.dayofweek.values.astype(int)
        state['age'] = (dates[view['cutoff_index'] - 1] - dates).days.values.astype(float)
    hist = np.asarray(view['target_history'], dtype=float)
    obs = np.asarray(view['target_observed'], dtype=bool)
    state['history'] = np.where(obs, hist, np.nan)
    return (view, state)

def engineer(prepared, card, state):
    hist = np.asarray(prepared['target_history'], dtype=float)
    obs = np.asarray(prepared['target_observed'], dtype=bool)
    clean = np.empty_like(hist)
    starts = []
    for d in range(hist.shape[0]):
        clean[d] = _log_bridge(hist[d], obs[d])
        starts.append(0 if not TRIM_LAUNCH else _dense_start(obs[d]))
    state['context_start'] = int(min(starts)) if starts else 0
    variables = dict(prepared)
    variables['target_history'] = clean
    return (variables, state)

def _yoy_known_row(clean, L, C, H, lag=YOY_LAG, weeks=(-7, 0, 7)):
    if L < lag + YOY_MIN_OVERLAP:
        return None
    base = np.arange(L - C, L + H) - lag
    acc = np.zeros(len(base))
    for w in weeks:
        acc += clean[np.clip(base + w, 0, L - 1)]
    row = acc / len(weeks)
    row[base < 0] = row[base >= 0][0] if np.any(base >= 0) else clean[0]
    return row

def select_context(variables, card, state):
    L = variables['cutoff_index']
    C = min(L - state.get('context_start', 0), card['limits']['max_context'])
    C = max(int(C), 1)
    H = variables['horizon']
    pf = variables['known_features']
    known = [pf[j, -(C + H):] for j in range(len(pf))]
    names = list(variables['known_names'])
    hist = np.asarray(variables['target_history'], dtype=float)
    if hist.shape[0] == 1:
        row = _yoy_known_row(hist[0], L, C, H)
        if row is not None and np.all(np.isfinite(row)):
            known.append(row)
            names.append('orders_annual_ref')
    kf = np.asarray(known, dtype=float) if known else None
    return ([{'target_indices': list(range(state['N'])), 'targets': hist[:, -C:], 'past_only': None, 'known_future': kf, 'past_names': [], 'known_names': names, 'provenance': 'Bridged targets, official known covariates + weekday-aligned annual reference'}], state)

def postprocess(outputs, cases, state):
    N, H = (state['N'], state['H'])
    nq = 9
    point = np.empty((N, H))
    quantiles = np.empty((N, H, nq))
    for output, case in zip(outputs, cases):
        point[case['target_indices']] = output['point']
        quantiles[case['target_indices']] = output['quantiles']
        nq = output['quantiles'].shape[-1]
    L = state['L']
    dow = state.get('dow')
    comps = {}
    if dow is not None and pd is not None:
        hist_dow = dow[:L][-DOW_LOOKBACK:]
        hist_age = state['age'][:L][-DOW_LOOKBACK:]
        fc_dow = dow[L:L + H]
        adj_all = np.zeros((N, H))
        for d in range(N):
            hv = state['history'][d][-DOW_LOOKBACK:]
            pv = point[d]
            if not np.all(np.isfinite(pv)) or np.any(pv <= 0):
                continue
            adj = _dow_adjustment(hv, hist_dow, hist_age, pv, fc_dow)
            if adj is None:
                continue
            adj_all[d] = adj
            f = np.exp(adj)
            point[d] *= f
            quantiles[d] *= f[:, None]
        comps['weekly_log_adjustment'] = adj_all
    a = _quantile_shrink(nq)
    med = quantiles[:, :, nq // 2:nq // 2 + 1]
    quantiles = med + a * (quantiles - med)
    point = med[:, :, 0]
    quantiles = np.sort(quantiles, axis=-1)
    return {'point': point, 'quantiles': quantiles, 'components': comps}

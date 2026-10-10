import numpy as np
import calendar_feats as cf
import specialist as sp
TFM_WEIGHT = 0.65
SPEC_WINDOWS = ((1095, 3.0, False), (1095, 5.0, True))

def _hourly_grid_ok(timestamps, L, H):
    import datetime as dt
    try:
        t0 = dt.datetime.strptime(str(timestamps[0])[:19], '%Y-%m-%dT%H:%M:%S')
        t1 = dt.datetime.strptime(str(timestamps[1])[:19], '%Y-%m-%dT%H:%M:%S')
        tl = dt.datetime.strptime(str(timestamps[L - 1])[:19], '%Y-%m-%dT%H:%M:%S')
    except Exception:
        return False
    if (t1 - t0).total_seconds() != 3600:
        return False
    return t0.hour == 0 and tl.hour == 23 and (L % 24 == 0) and (H == 24)

def fit_specialist(view, state, oos_days=660):
    L, H = (state['L'], state['H'])
    ts = list(view['timestamps'])
    if not _hourly_grid_ok(ts, L, H):
        return (None, None)
    known = np.asarray(view['known_features'], float)
    if known.shape[0] < 2:
        return (None, None)
    y = np.asarray(view['target_history'], float)[0]
    obs = np.asarray(view.get('target_observed', np.ones_like(y)), float)
    if obs.ndim > 1:
        obs = obs[0]
    y = np.where(obs > 0.5, y, np.nan)
    if np.isnan(y).mean() > 0.2:
        return (None, None)
    price = np.concatenate([y, np.full(H, np.nan)])
    P, S, C = sp.build_daily(price, known[0][:L + H], known[1][:L + H])
    if len(P) < 400:
        return (None, None)
    hours, wdays, dates = cf.parse_times(ts[::24][-len(P):])
    offd, hol = cf.offday_flag(wdays, dates)
    wd = np.asarray(wdays)
    hol_flag = np.array([1.0 if d in hol else 0.0 for d in dates])
    doy = np.array([d.timetuple().tm_yday for d in dates], float)
    X, _ = sp.feature_tensor(P, S, C, wd, hol_flag)
    X = sp.add_seasonal(X, doy)
    origin = len(P) - 1
    preds = []
    for td, ridge, rob in SPEC_WINDOWS:
        q = sp.lear_predict(X, P, origin, train_days=td, ridge=ridge, robust=rob)
        if q is not None and np.isfinite(q).all():
            preds.append(q)
    if not preds:
        return (None, None)
    out = np.mean(preds, 0)
    recent = P[-365:]
    lo, hi = (np.nanmin(recent), np.nanmax(recent))
    span = hi - lo
    out = np.clip(out, lo - 0.5 * span, hi + 0.5 * span)
    hist = None
    try:
        first, ser = sp.rolling_oos_series(X, P, origin, oos_days, step=28)
        full = np.full(origin, np.nan)
        full_mat = np.full((origin, 24), np.nan)
        full_mat[first:origin] = ser
        hist = np.clip(full_mat.reshape(-1), lo - 0.5 * span, hi + 0.5 * span)
    except Exception:
        hist = None
    return (out, hist)

def preprocess(view, card):
    state = {'N': len(view['target_ids']), 'H': int(view['horizon']), 'L': int(view['cutoff_index'])}
    return (view, state)

def _attach_specialist(view, state):
    try:
        spec, hist = fit_specialist(view, state)
    except Exception:
        spec, hist = (None, None)
    state['spec'] = None if spec is None else np.asarray(spec, float)
    state['spec_hist'] = hist
    return state

def engineer(prepared, card, state):
    v = dict(prepared)
    L, H = (state['L'], state['H'])
    ts = list(v['timestamps'])
    hours, wdays, dates = cf.parse_times(ts)
    offday, _ = cf.offday_flag(wdays, dates)
    y = np.asarray(v['target_history'], float)[0]
    prof, _tab = cf.causal_calendar_profile(y, L, hours, wdays, offday, weeks=8)
    _attach_specialist(v, state)
    known = np.asarray(v['known_features'], float)
    names = list(v['known_names'])
    extra, extra_names = ([], [])
    if known.shape[0] >= 1:
        sysl = known[0]
        extra.append(cf.trailing_hourly_norm(sysl, L, 28))
        extra_names.append('sysload_anomaly_28d')
        if known.shape[0] >= 2:
            comed = known[1]
            extra.append(comed / np.maximum(np.abs(sysl), 1.0))
            extra_names.append('comed_share')
    extra.append(offday)
    extra_names.append('offday')
    extra.append(prof)
    extra_names.append('daytype_hour_price_profile')
    spec, hist = (state.get('spec'), state.get('spec_hist'))
    if spec is not None:
        est = np.array(prof, float)
        if hist is not None:
            m = np.isfinite(hist)
            est[:L][m[:L]] = hist[:L][m[:L]]
        est[L:L + H] = spec
        extra.append(est)
        extra_names.append('lear_dayahead_estimate')
    add = np.asarray(extra, float)
    add = np.nan_to_num(add, nan=0.0, posinf=0.0, neginf=0.0)
    v['known_features'] = np.vstack([known, add]) if known.size else add
    v['known_names'] = names + extra_names
    state['extra_names'] = extra_names
    return (v, state)

def select_context(variables, card, state):
    C = min(int(variables['cutoff_index']), card['limits']['max_context'])
    H = state['H']
    po = np.asarray(variables['past_features'], float)
    pf = np.asarray(variables['known_features'], float)
    case = {'target_indices': list(range(state['N'])), 'targets': np.asarray(variables['target_history'], float)[:, -C:], 'past_only': po[:, -C:] if po.size else None, 'known_future': pf[:, -(C + H):] if pf.size else None, 'past_names': list(variables['past_names']), 'known_names': list(variables['known_names']), 'provenance': 'All targets, official known load forecasts plus four causal calendar/operating-state covariates, max_context=15360'}
    return ([case], state)

def postprocess(outputs, cases, state):
    point = np.empty((state['N'], state['H']))
    quantiles = np.empty((state['N'], state['H'], 9))
    for output, case in zip(outputs, cases):
        point[case['target_indices']] = output['point']
        quantiles[case['target_indices']] = output['quantiles']
    spec = state.get('spec')
    if spec is not None and state['N'] == 1 and (spec.shape[0] == state['H']):
        native_point = point[0].copy()
        blended = TFM_WEIGHT * native_point + (1.0 - TFM_WEIGHT) * spec
        shift = blended - native_point
        point[0] = blended
        quantiles[0] = quantiles[0] + shift[:, None]
        quantiles[0] = np.sort(quantiles[0], axis=-1)
        state['blend_shift'] = shift
    return {'point': point, 'quantiles': quantiles, 'components': {'specialist_used': spec is not None}}

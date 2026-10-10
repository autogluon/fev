import numpy as np
import calendar_feats as cf
import specialist as sp
import gbm_expert as gx
SPEC_WINDOWS = ((1095, 3.0, False), (1095, 5.0, True))
GBM_W = 0.5
AUX_OFFSETS = (24, 48, 72, 96, 120)
ACTIONS = np.array([1.0, 0.85, 0.7, 0.55, 0.4, 0.25, 0.1, 0.0])
WIDTHS = np.array([0.85, 1.0, 1.15, 1.3])
WIDTH_PEN = 0.015
STD_PEN = 0.35
RAW_PREF = 0.02

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
        return (None, None, False)
    known = np.asarray(view['known_features'], float)
    if known.shape[0] < 2:
        return (None, None, False)
    y = np.asarray(view['target_history'], float)[0]
    obs = np.asarray(view.get('target_observed', np.ones_like(y)), float)
    if obs.ndim > 1:
        obs = obs[0]
    y = np.where(obs > 0.5, y, np.nan)
    if np.isnan(y).mean() > 0.2:
        return (None, None, False)
    price = np.concatenate([y, np.full(H, np.nan)])
    P, S, C = sp.build_daily(price, known[0][:L + H], known[1][:L + H])
    if len(P) < 400:
        return (None, None, False)
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
        return (None, None, False)
    out = np.mean(preds, 0)
    recent = P[-365:]
    lo, hi = (np.nanmin(recent), np.nanmax(recent))
    span = hi - lo
    clip = lambda a: np.clip(a, lo - 0.5 * span, hi + 0.5 * span)
    out = clip(out)
    hist = None
    try:
        first, ser = sp.rolling_oos_series(X, P, origin, oos_days, step=28)
        full_mat = np.full((origin, 24), np.nan)
        full_mat[first:origin] = ser
        hist = clip(full_mat)
    except Exception:
        hist = None
    gbm_used = False
    try:
        g_today, g_hist = gx.gbm_forecast(X, P, origin, oos=3)
    except Exception:
        g_today, g_hist = (None, None)
    if g_today is not None and np.isfinite(g_today).all():
        out = (1.0 - GBM_W) * out + GBM_W * clip(np.asarray(g_today, float))
        gbm_used = True
        if hist is not None and g_hist is not None:
            gh = clip(np.asarray(g_hist, float))
            for j, d in enumerate(range(origin - len(gh), origin)):
                if np.isfinite(gh[j]).all() and np.isfinite(hist[d]).all():
                    hist[d] = (1.0 - GBM_W) * hist[d] + GBM_W * gh[j]
    return (out, hist, gbm_used)

def preprocess(view, card):
    state = {'N': len(view['target_ids']), 'H': int(view['horizon']), 'L': int(view['cutoff_index'])}
    return (view, state)
MIN_PROFILE_DAYS = 21

def engineer(prepared, card, state):
    v = dict(prepared)
    try:
        spec, hist, gbm_used = fit_specialist(v, state)
    except Exception:
        spec, hist, gbm_used = (None, None, False)
    state['spec'] = None if spec is None else np.asarray(spec, float)
    state['gbm_used'] = gbm_used
    state['spec_hist'] = hist
    state['y'] = np.asarray(v['target_history'], float)[0]
    state['cov_attached'] = False
    L, H = (state['L'], state['H'])
    maxctx = int(card.get('limits', {}).get('max_context', 15360))
    if L > maxctx or not _hourly_grid_ok(list(v['timestamps']), L, H):
        return (v, state)
    ts = list(v['timestamps'])
    hours, wdays, dates = cf.parse_times(ts)
    offday, _ = cf.offday_flag(wdays, dates)
    y = state['y']
    warm = L >= 24 * MIN_PROFILE_DAYS
    known = np.asarray(v['known_features'], float)
    names = list(v['known_names'])
    extra, extra_names = ([], [])
    if warm and known.shape[0] >= 1:
        sysl = known[0]
        extra.append(cf.trailing_hourly_norm(sysl, L, 28))
        extra_names.append('sysload_anomaly_28d')
        if known.shape[0] >= 2:
            extra.append(known[1] / np.maximum(np.abs(sysl), 1.0))
            extra_names.append('comed_share')
    extra.append(offday)
    extra_names.append('offday')
    if warm:
        prof, _tab = cf.causal_calendar_profile(y, L, hours, wdays, offday, weeks=8)
        extra.append(prof)
        extra_names.append('daytype_hour_price_profile')
        if state['spec'] is not None and hist is not None:
            est = np.array(prof, float)
            hflat = np.asarray(hist, float).reshape(-1)
            m = np.isfinite(hflat)
            est[:len(hflat)][m] = hflat[m]
            est[L:L + H] = state['spec']
            extra.append(est)
            extra_names.append('expert_dayahead_estimate')
    add = np.nan_to_num(np.asarray(extra, float), nan=0.0, posinf=0.0, neginf=0.0)
    v['known_features'] = np.vstack([known, add]) if known.size else add
    v['known_names'] = names + extra_names
    state['cov_attached'] = True
    state['cov_names'] = extra_names
    return (v, state)

def _case(v, state, offset, aux=False):
    L, H = (state['L'], state['H'])
    end = L - offset
    C = min(end, 15360)
    po = np.asarray(v['past_features'], float)
    pf = np.asarray(v['known_features'], float)
    row = {'target_indices': list(range(state['N'])), 'targets': np.asarray(v['target_history'], float)[:, end - C:end], 'past_only': po[:, end - C:end] if po.size else None, 'known_future': pf[:, end - C:end + H] if pf.size else None, 'past_names': list(v['past_names']), 'known_names': list(v['known_names']), 'provenance': {'arm': 'native-bare', 'offset': offset}}
    if aux:
        row['auxiliary'] = True
        row['origin_offset'] = offset
    return row

def select_context(variables, card, state):
    v = variables
    L, H = (state['L'], state['H'])
    cases = [_case(v, state, 0)]
    state['aux'] = []
    if state.get('spec') is not None and state.get('spec_hist') is not None and (L % 24 == 0):
        y = state['y']
        for off in AUX_OFFSETS:
            if L - off < 2 * H:
                continue
            day = (L - off) // 24
            lear_day = state['spec_hist'][day] if day < len(state['spec_hist']) else None
            truth_day = y[L - off:L - off + H]
            if lear_day is None or not np.isfinite(lear_day).all() or (not np.isfinite(truth_day).all()):
                continue
            cases.append(_case(v, state, off, aux=True))
            state['aux'].append({'offset': off, 'lear': np.asarray(lear_day, float), 'truth': np.asarray(truth_day, float)})
    return (cases, state)

def postprocess(outputs, cases, state):
    N, H = (state['N'], state['H'])
    point = np.asarray(outputs[0]['point'], float).reshape(N, H).copy()
    quantiles = np.asarray(outputs[0]['quantiles'], float).reshape(N, H, 9).copy()
    spec = state.get('spec')
    gate = {'w': 1.0, 's': 1.0, 'days': 0, 'native_mae': None, 'lear_mae': None}
    if spec is not None and N == 1 and (len(state.get('aux', [])) >= 2):
        days = state['aux']
        levels = np.arange(1, 10) / 10.0
        losses = []
        for info, o in zip(days, outputs[1:]):
            np_pt = np.asarray(o['point'], float).reshape(H)
            nq = np.asarray(o['quantiles'], float).reshape(H, 9)
            med = nq[:, 4:5]
            row = []
            for w in ACTIONS:
                shift = (1.0 - w) * (info['lear'] - np_pt)
                srow = []
                for s in WIDTHS:
                    qq = np.sort(med + s * (nq - med) + shift[:, None], axis=-1)
                    e = info['truth'][:, None] - qq
                    srow.append(np.mean(2 * np.where(e >= 0, levels * e, (levels - 1) * e)))
                row.append(srow)
            losses.append(row)
        loss = np.array(losses)
        i1 = int(np.argmin(np.abs(WIDTHS - 1.0)))
        base = np.maximum(loss[:, 0, i1].mean(), 1e-09)
        rel = loss / base
        obj = rel.mean(0) + STD_PEN * rel.std(0) / np.sqrt(len(loss)) + RAW_PREF * (1.0 - ACTIONS)[:, None] + WIDTH_PEN * (np.abs(WIDTHS - 1.0) / 0.15)[None, :]
        ai, bi = np.unravel_index(int(np.argmin(obj)), obj.shape)
        w, s = (float(ACTIONS[ai]), float(WIDTHS[bi]))
        gate = {'w': w, 's': s, 'days': len(loss), 'native_mae': float(loss[:, 0, i1].mean()), 'lear_mae': float(np.mean([np.mean(np.abs(d['lear'] - d['truth'])) for d in days])), 'objective': [[float(x) for x in r] for r in obj]}
        if (w < 1.0 or s != 1.0) and spec.shape[0] == H:
            native_point = point[0].copy()
            blended = w * native_point + (1.0 - w) * spec
            shift = blended - native_point
            med = quantiles[0][:, 4:5]
            point[0] = blended
            quantiles[0] = np.sort(med + s * (quantiles[0] - med) + shift[:, None], axis=-1)
    return {'point': point, 'quantiles': quantiles, 'components': {'mechanism': 'pinball-gated native vs LEAR+LGBM committee, bare inputs', 'gate': gate, 'aux_offsets_used': [a['offset'] for a in state.get('aux', [])], 'lear_available': spec is not None, 'gbm_in_committee': state.get('gbm_used', False), 'context_poor_covariates': state.get('cov_attached', False), 'covariate_names': state.get('cov_names')}}

import numpy as np
import calendar_feats as cf
import specialist as sp
import gbm_expert as gx
SPEC_WINDOWS = ((1095, 3.0, False), (1095, 5.0, True))
GBM_W = 0.5
AUX_OFFSETS = (24, 48, 72, 96, 120)
ACTIONS = np.array([1.0, 0.85, 0.7, 0.55, 0.4, 0.25, 0.1, 0.0])
WIDTHS = np.array([0.85, 1.0, 1.15, 1.3])
SPAIRS = np.array([(1.0, 1.0), (0.85, 1.0), (1.0, 0.85), (1.15, 1.0), (1.0, 1.15), (1.3, 1.0), (1.0, 1.3), (0.85, 0.85), (1.15, 1.15)])
GAINS = np.array([0.0, 0.25, 0.5])
ANCHOR_DEF = 0.25
ANCHOR_PEN = 0.008
ANCHOR_MIN_DAYS = 8
WIDTH_PEN = 0.015
STD_PEN = 0.35
RAW_PREF = 0.02

def _ratio_env(y, load, day, horizon_load):
    if day < ANCHOR_MIN_DAYS:
        return None
    P = y[24 * (day - 7):24 * day].reshape(7, 24)
    Ld = load[24 * (day - 7):24 * day].reshape(7, 24)
    if not (np.isfinite(P).all() and np.isfinite(Ld).all()):
        return None
    dmr = P.mean(1) / np.maximum(Ld.mean(1), 1.0)
    fl = float(np.mean(horizon_load))
    if not np.isfinite(fl) or fl <= 0 or (not np.isfinite(dmr).all()):
        return None
    return {'anchor': float(np.median(dmr) * fl), 'floor': float(np.min(dmr)), 'fload': fl, 'lmax': float(np.max(Ld.mean(1)))}

def _floor_clamp(env, np_pt):
    if env is None:
        return 0.0
    m = float(np.mean(np_pt))
    if m / env['fload'] < env['floor']:
        sh = env['floor'] * env['fload'] - m
        cap = 0.35 * abs(m) + 3.0
        return float(np.clip(sh, 0.0, cap))
    return 0.0
SURGE_EXC = 1.03

def _anchor_shift(env, native_mean, g):
    if env is None or g <= 0.0:
        return 0.0
    sh = g * (env['anchor'] - native_mean)
    if sh < 0.0 and env['fload'] > SURGE_EXC * env['lmax']:
        return 0.0
    cap = 0.25 * abs(native_mean) + 2.0
    return float(np.clip(sh, -cap, cap))

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

    def _members_at(org):
        ms = []
        short = None
        pair = []
        for td, ridge, rob in SPEC_WINDOWS:
            q = sp.lear_predict(X, P, org, train_days=td, ridge=ridge, robust=rob)
            if q is not None and np.isfinite(q).all():
                pair.append(q)
        if pair:
            ms.append(np.mean(pair, 0))
        for td, ridge, rob in ((364, 3.0, False), (84, 5.0, True)):
            q = sp.lear_predict(X, P, org, train_days=td, ridge=ridge, robust=rob)
            if q is not None and np.isfinite(q).all():
                ms.append(q)
                if td == 84:
                    short = q
        return (ms, short)
    members, short_today = _members_at(origin)
    if not members:
        return (None, None, False)
    recent = P[-365:]
    lo, hi = (np.nanmin(recent), np.nanmax(recent))
    span = hi - lo
    clip = lambda a: np.clip(a, lo - 0.5 * span, hi + 0.5 * span)
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
        g_today, g_hist = gx.gbm_forecast(X, P, origin, oos=5)
    except Exception:
        g_today, g_hist = (None, None)
    if g_today is not None and np.isfinite(g_today).all():
        members.append(np.asarray(g_today, float))
        gbm_used = True
    out = clip(np.mean(members, 0))
    short_out = None if short_today is None else clip(np.asarray(short_today, float))
    oos_map = {}
    oos_short = {}
    for off in AUX_OFFSETS:
        day = origin - off // 24
        if day < 420:
            continue
        ms, sh = _members_at(day)
        if sh is not None and np.isfinite(sh).all():
            oos_short[day] = clip(np.asarray(sh, float))
        if g_hist is not None:
            j = day - (origin - len(g_hist))
            if 0 <= j < len(g_hist) and np.isfinite(g_hist[j]).all():
                ms.append(np.asarray(g_hist[j], float))
        if ms:
            oos_map[day] = clip(np.mean(ms, 0))
    if hist is not None:
        for day, val in oos_map.items():
            if np.isfinite(val).all():
                hist[day] = val
    return (out, hist, gbm_used, short_out, oos_short)

def preprocess(view, card):
    state = {'N': len(view['target_ids']), 'H': int(view['horizon']), 'L': int(view['cutoff_index'])}
    return (view, state)
MIN_PROFILE_DAYS = 21

def engineer(prepared, card, state):
    v = dict(prepared)
    try:
        spec, hist, gbm_used, spec_short, oos_short = fit_specialist(v, state)
    except Exception:
        spec, hist, gbm_used, spec_short, oos_short = (None, None, False, None, {})
    state['spec'] = None if spec is None else np.asarray(spec, float)
    state['spec_short'] = None if spec_short is None else np.asarray(spec_short, float)
    state['oos_short'] = oos_short or {}
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
    state['env_today'] = None
    hourly = _hourly_grid_ok(list(v['timestamps']), L, H)
    known = np.asarray(v['known_features'], float)
    load = known[0][:L] if known.shape[0] >= 1 and known.shape[1] >= L + H else None
    if hourly and load is not None and (L % 24 == 0):
        y = state['y']
        state['env_today'] = _ratio_env(y, load, L // 24, known[0][L:L + H])
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
            env = None
            if load is not None:
                env = _ratio_env(y, load, day, load[24 * day:24 * day + 24])
            cases.append(_case(v, state, off, aux=True))
            short_day = state.get('oos_short', {}).get(day)
            state['aux'].append({'offset': off, 'lear': np.asarray(lear_day, float), 'truth': np.asarray(truth_day, float), 'short': None if short_day is None else np.asarray(short_day, float), 'env': env})
    return (cases, state)

def postprocess(outputs, cases, state):
    N, H = (state['N'], state['H'])
    point = np.asarray(outputs[0]['point'], float).reshape(N, H).copy()
    quantiles = np.asarray(outputs[0]['quantiles'], float).reshape(N, H, 9).copy()
    spec = state.get('spec')
    env_today = state.get('env_today')
    anchor_today = None if env_today is None else env_today['anchor']
    clamp_today = 0.0
    if N == 1:
        clamp_today = _floor_clamp(env_today, point[0])
        if clamp_today > 0.0:
            point[0] = point[0] + clamp_today
            quantiles[0] = quantiles[0] + clamp_today
    gate = {'w': 1.0, 's': 1.0, 'g': 0.0, 'days': 0, 'native_mae': None, 'lear_mae': None}
    gated = False
    if clamp_today > 0.0 and N == 1:
        med = quantiles[0][:, 4:5]
        dq = quantiles[0] - med
        quantiles[0] = np.sort(med + np.where(dq > 0, 1.3 * dq, dq), axis=-1)
        gate = {'w': 1.0, 's': 'asym1.3up', 'g': 0.0, 'days': 0, 'native_mae': None, 'lear_mae': None, 'clamp_override': True}
    elif spec is not None and N == 1 and (len(state.get('aux', [])) >= 2):
        days = state['aux']
        spec_short = state.get('spec_short')
        n_exp = 2 if spec_short is not None and sum((1 for d in days if d.get('short') is not None)) >= 2 else 1
        levels = np.arange(1, 10) / 10.0
        losses = []
        for info, o in zip(days, outputs[1:]):
            np_pt = np.asarray(o['point'], float).reshape(H)
            nq = np.asarray(o['quantiles'], float).reshape(H, 9)
            cl = _floor_clamp(info.get('env'), np_pt)
            np_pt = np_pt + cl
            nq = nq + cl
            med = nq[:, 4:5]
            env_day = info.get('env')
            erow = []
            for ei in range(n_exp):
                xser = info['lear'] if ei == 0 else info['short'] if info.get('short') is not None else info['lear']
                row = []
                for w in ACTIONS:
                    srow = []
                    for sl, su in SPAIRS:
                        grow = []
                        dq0 = nq - med
                        dqs = np.where(dq0 > 0, su * dq0, sl * dq0)
                        for g in GAINS:
                            gsh = _anchor_shift(env_day, float(np_pt.mean()), g)
                            tsh = w * gsh + (1.0 - w) * (xser - np_pt)
                            qq = np.sort(med + dqs + tsh[:, None], axis=-1)
                            e = info['truth'][:, None] - qq
                            grow.append(np.mean(2 * np.where(e >= 0, levels * e, (levels - 1) * e)))
                        srow.append(grow)
                    row.append(srow)
                erow.append(row)
            losses.append(erow)
        loss = np.array(losses)
        i1 = 0
        base = np.maximum(loss[:, 0, 0, i1, 0].mean(), 1e-09)
        rel = loss / base
        spen = (np.abs(SPAIRS[:, 0] - 1.0) + np.abs(SPAIRS[:, 1] - 1.0)) / 0.15
        SHORT_PEN = 0.004
        obj = rel.mean(0) + STD_PEN * rel.std(0) / np.sqrt(len(loss)) + RAW_PREF * (1.0 - ACTIONS)[None, :, None, None] + WIDTH_PEN * spen[None, None, :, None] + ANCHOR_PEN * (GAINS / 0.25)[None, None, None, :] + SHORT_PEN * np.arange(loss.shape[1])[:, None, None, None]
        ei, ai, bi, ci = np.unravel_index(int(np.argmin(obj)), obj.shape)
        w, g = (float(ACTIONS[ai]), float(GAINS[ci]))
        sl, su = (float(SPAIRS[bi][0]), float(SPAIRS[bi][1]))
        gate = {'w': w, 's': [sl, su], 'g': g, 'expert': 'short84' if ei == 1 else 'committee', 'days': len(loss), 'native_mae': float(loss[:, 0, 0, i1, 0].mean()), 'lear_mae': float(np.mean([np.mean(np.abs(d['lear'] - d['truth'])) for d in days])), 'anchor_days': int(sum((1 for d in days if d.get('env') is not None)))}
        if (w < 1.0 or sl != 1.0 or su != 1.0 or (g > 0.0)) and spec.shape[0] == H:
            native_point = point[0].copy()
            xf = spec if ei == 0 or state.get('spec_short') is None else state['spec_short']
            gsh = _anchor_shift(env_today, float(native_point.mean()), g)
            tsh = w * gsh + (1.0 - w) * (xf - native_point)
            med = quantiles[0][:, 4:5]
            dq0 = quantiles[0] - med
            dqs = np.where(dq0 > 0, su * dq0, sl * dq0)
            point[0] = native_point + tsh
            quantiles[0] = np.sort(med + dqs + tsh[:, None], axis=-1)
            gate['anchor_shift_today'] = float(gsh)
        gated = True
    elif anchor_today is not None and N == 1 and (clamp_today == 0.0):
        gsh = _anchor_shift(env_today, float(point[0].mean()), ANCHOR_DEF)
        if gsh != 0.0:
            point[0] = point[0] + gsh
            quantiles[0] = quantiles[0] + gsh
            gate = {'w': 1.0, 's': 1.0, 'g': ANCHOR_DEF, 'days': 0, 'native_mae': None, 'lear_mae': None, 'anchor_shift_today': float(gsh), 'default_pull': True}
    return {'point': point, 'quantiles': quantiles, 'components': {'mechanism': 'low-ratio-floor clamp + pinball-gated native vs LEAR+LGBM committee + load-anchor level pull', 'gate': gate, 'gate_evidence_used': gated, 'clamp_today': clamp_today, 'anchor_today': anchor_today, 'aux_offsets_used': [a['offset'] for a in state.get('aux', [])], 'lear_available': spec is not None, 'gbm_in_committee': state.get('gbm_used', False), 'context_poor_covariates': state.get('cov_attached', False), 'covariate_names': state.get('cov_names')}}

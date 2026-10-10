import importlib.util
import pathlib
import numpy as np

def _load(name):
    path = pathlib.Path(__file__).with_name(name + '.py')
    spec = importlib.util.spec_from_file_location('epf_' + name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
CB = _load('calendar_be')
SB = _load('specialist_be')
CONFIG = {'primary_arm': 'winsor', 'context': 15360, 'profile_weeks': 8, 'anom_days': 28, 'spec_train_days': 1095, 'spec_hist_days': 660, 'spec_step': 30, 'aux_offsets': [24, 72, 168, 336, 504, 720, 1008, 1344, 1680, 2016, 2520, 3024, 3528, 4200, 5040, 5880, 6720, 7560, 8400, 9240, 10920, 12600, 14280, 16800], 'aux_arms': ['ctl', 'winsor', 'deseas'], 'winsor_lo': 0.5, 'winsor_hi': 99.5, 'winsor_protect': 168}
ARMS = {'ctl': (False, False, False, 'none'), 'cal': (True, False, False, 'none'), 'anom': (False, True, False, 'none'), 'all': (True, True, False, 'none'), 'spec': (False, False, True, 'none'), 'calspec': (True, False, True, 'none'), 'winsor': (False, False, False, 'winsor'), 'deseas': (False, False, False, 'deseas'), 'wdeseas': (False, False, False, 'wdeseas')}

def _transform(block, state, origin, kind):
    H = state['H']
    out = np.array(block, float, copy=True)
    shift = np.zeros((out.shape[0], H))
    if kind in ('deseas', 'wdeseas'):
        prof = CB.daytype_hour_profile(state['hist'][0], origin, state['hour'], state['offday'], CONFIG['profile_weeks'])
        C = out.shape[1]
        out = out - prof[origin - C:origin][None, :]
        shift = np.repeat(prof[origin:origin + H][None, :], out.shape[0], 0)
    if kind in ('winsor', 'wdeseas'):
        prot = CONFIG['winsor_protect']
        for d in range(out.shape[0]):
            row = out[d]
            if row.size < 3 * prot:
                continue
            hi = np.percentile(row, CONFIG['winsor_hi'])
            lo = np.percentile(row, CONFIG['winsor_lo'])
            rec = row[-prot:]
            hi = max(hi, float(rec.max()))
            lo = min(lo, float(rec.min()))
            if hi > lo:
                out[d] = np.clip(row, lo, hi)
    return (out, shift)

def preprocess(view, card):
    hist = np.asarray(view['target_history'], dtype=float)
    obs = np.asarray(view['target_observed'])
    bad = ~obs.astype(bool) | ~np.isfinite(hist) if obs.shape == hist.shape else ~np.isfinite(hist)
    if bad.any():
        for d in range(hist.shape[0]):
            row, idx = (hist[d], np.where(~bad[d])[0])
            row[:] = np.interp(np.arange(row.size), idx, row[idx]) if idx.size else 0.0
    known = np.asarray(view['known_features'], dtype=float) if np.size(view['known_features']) else np.zeros((0, 0))
    if known.size and (not np.all(np.isfinite(known))):
        for j in range(known.shape[0]):
            r = known[j]
            idx = np.where(np.isfinite(r))[0]
            r[:] = np.interp(np.arange(r.size), idx, r[idx]) if idx.size else 0.0
    prepared = dict(view)
    prepared['target_history'] = hist
    prepared['known_features'] = known
    state = {'N': int(hist.shape[0]), 'H': int(view['horizon']), 'L': int(view['cutoff_index']), 'hist': hist, 'aux': [], 'notes': []}
    return (prepared, state)

def engineer(prepared, card, state):
    ts = list(prepared['timestamps'])
    _, hour, dow, _days = CB.time_axis(ts)
    state['hour'] = hour
    state['offday'] = CB.offday_flag(ts)
    state['dow'] = dow
    days = np.asarray([str(t)[:10] for t in ts], dtype='datetime64[D]')
    state['doy'] = ((days - days.astype('datetime64[Y]')).astype(int) + 1).astype(float)
    state['known_names'] = list(prepared['known_names']) if np.size(prepared['known_names']) else []
    known = prepared['known_features']
    lower = [str(n).lower() for n in state['known_names']]
    state['load_idx'] = next((i for i, n in enumerate(lower) if 'load' in n), None)
    state['gen_idx'] = next((i for i, n in enumerate(lower) if 'generation' in n), None)
    state['notes'].append('calendar axis built; load column index=%s' % state['load_idx'])
    state['known'] = known
    return (prepared, state)

def _expert_variate(state, origin):
    key = int(origin)
    if key in state.setdefault('spec_cache', {}):
        return state['spec_cache'][key]
    out = None
    try:
        hour = state['hour']
        if hour[0] != 0 or origin % 24 != 0:
            raise ValueError('context is not day aligned')
        known = state['known']
        n_hours = state['hour'].shape[0]
        endd = origin // 24 + 1
        ndays = min(CONFIG['spec_train_days'] + CONFIG['spec_hist_days'] + 30, endd)
        sl = slice((endd - ndays) * 24, endd * 24)
        price = np.concatenate([state['hist'][0], np.full(max(0, n_hours - state['hist'].shape[1]), np.nan)])
        P = price[sl].reshape(-1, 24).copy()
        P[-1] = np.nan
        G = known[state['gen_idx']][sl].reshape(-1, 24)
        S = known[state['load_idx']][sl].reshape(-1, 24)
        dow = state['dow'][sl][::24]
        offd = state['offday'][sl][::24]
        doy = state['doy'][sl][::24]
        X, _ = SB.feature_tensor(P, G, S, dow, offd, doy)
        od = P.shape[0] - 1
        fc = SB.lear_predict(X, P, od, train_days=CONFIG['spec_train_days'], ridge=3.0)
        fcr = SB.lear_predict(X, P, od, train_days=CONFIG['spec_train_days'], ridge=5.0, robust=True)
        parts = [a for a in (fc, fcr) if a is not None and np.isfinite(a).all()]
        if not parts:
            raise ValueError('specialist failed')
        fc = np.mean(parts, 0)
        first, ser = SB.rolling_oos_series(X, P, od, CONFIG['spec_hist_days'], train_days=CONFIG['spec_train_days'], step=CONFIG['spec_step'])
        prof = CB.daytype_hour_profile(state['hist'][0], origin, state['hour'], state['offday'], CONFIG['profile_weeks'])
        est = prof.copy()
        base_day = endd - ndays
        hi = np.nanpercentile(P[max(0, od - 365):od], 99.5)
        lo = np.nanpercentile(P[max(0, od - 365):od], 0.5)
        span = hi - lo
        ser = np.clip(ser, lo - 0.75 * span, hi + 0.75 * span)
        a = (base_day + first) * 24
        b = a + ser.shape[0] * 24
        flat = ser.reshape(-1)
        good = np.isfinite(flat)
        seg = est[a:b]
        seg[good] = flat[good]
        est[a:b] = seg
        est[origin:origin + state['H']] = np.clip(fc, lo - 0.75 * span, hi + 0.75 * span)
        out = est
    except Exception as exc:
        state['notes'].append('expert variate failed at origin %d: %r' % (origin, exc))
        out = None
    state['spec_cache'][key] = out
    return out

def _extra_block(state, origin, use_cal, use_anom, use_spec=False):
    cols, names = ([], [])
    if use_cal:
        cols.append(state['offday'])
        names.append('offday')
        cols.append(CB.daytype_hour_profile(state['hist'][0], origin, state['hour'], state['offday'], CONFIG['profile_weeks']))
        names.append('daytype_hour_price_profile')
    if use_anom and state['load_idx'] is not None and np.size(state['known']):
        cols.append(CB.trailing_same_hour_ratio(state['known'][state['load_idx']], origin, CONFIG['anom_days']))
        names.append('sysload_anom_%dd' % CONFIG['anom_days'])
    if use_spec:
        est = _expert_variate(state, origin)
        if est is not None:
            cols.append(est)
            names.append('lear_dayahead_estimate')
    if not cols:
        return (None, [])
    return (np.asarray(cols, float), names)

def _make_case(state, origin, C, arm, auxiliary, offset, tag):
    H = state['H']
    block = np.ascontiguousarray(state['hist'][:, origin - C:origin], dtype=float)
    block = np.nan_to_num(block, nan=0.0, posinf=0.0, neginf=0.0)
    use_cal, use_anom, use_spec, kind = ARMS[arm]
    block, shift = _transform(block, state, origin, kind)
    block = np.ascontiguousarray(np.nan_to_num(block, nan=0.0, posinf=0.0, neginf=0.0))
    extra, extra_names = _extra_block(state, origin, use_cal, use_anom, use_spec)
    known = state['known']
    parts, names = ([], [])
    if np.size(known):
        parts.append(known)
        names += list(state['known_names'])
    if extra is not None:
        parts.append(extra)
        names += extra_names
    kf = None
    if parts:
        full = np.vstack(parts)
        kf = np.ascontiguousarray(full[:, origin - C:origin + H], dtype=float)
        kf = np.nan_to_num(kf, nan=0.0, posinf=0.0, neginf=0.0)
    prov = 'tag=%s arm=%s C=%d offset=%d origin=%d nvar=%d kind=%s' % (tag, arm, C, offset, origin, 0 if kf is None else kf.shape[0], kind)
    state.setdefault('shifts', {})[tag, arm, int(offset)] = shift
    case = {'target_indices': list(range(state['N'])), 'targets': block, 'past_only': None, 'known_future': kf, 'past_names': [], 'known_names': names if kf is not None else [], 'provenance': prov}
    if auxiliary:
        case['auxiliary'] = True
        case['origin_offset'] = int(offset)
    return case

def select_context(variables, card, state):
    H, L = (state['H'], state['L'])
    limit = int(card.get('limits', {}).get('max_context', 15360))

    def clamp(c, origin):
        c = int(min(c, origin, limit))
        return c - c % 24
    C = clamp(CONFIG['context'], L)
    cases = [_make_case(state, L, C, CONFIG['primary_arm'], False, 0, 'primary')]
    state['C'] = C
    hist = state['hist']
    for off in CONFIG['aux_offsets']:
        off = int(off)
        if off < H:
            continue
        origin = L - off
        if origin < 24 * 400:
            continue
        truth = hist[:, origin:origin + H]
        if not np.all(np.isfinite(truth)):
            continue
        Caux = clamp(CONFIG['context'], origin)
        if Caux < 24 * 30:
            continue
        for arm in CONFIG['aux_arms']:
            cases.append(_make_case(state, origin, Caux, arm, True, off, 'aux'))
            state['aux'].append({'pos': len(cases) - 1, 'arm': arm, 'offset': off, 'origin': origin, 'truth': truth.copy()})
    state['notes'].append('cases=%d (1 primary + %d auxiliary)' % (len(cases), len(cases) - 1))
    return (cases, state)

def _scale(hist):
    y = hist[0]
    return float(np.abs(y[24:] - y[:-24]).mean()) or 1.0

def postprocess(outputs, cases, state):
    N, H = (state['N'], state['H'])
    point = np.empty((N, H))
    quant = np.empty((N, H, 9))
    filled = np.zeros(N, dtype=bool)
    qlev = np.arange(1, 10) / 10.0
    sc = _scale(state['hist'])
    per_arm = {}
    detail = []
    for rec in state.get('aux', []):
        pos = rec['pos']
        if pos >= len(outputs):
            continue
        p = np.asarray(outputs[pos]['point'], float)
        q = np.asarray(outputs[pos]['quantiles'], float)
        sh = state.get('shifts', {}).get(('aux', rec['arm'], int(rec['offset'])))
        if sh is not None and np.any(sh):
            p = p + sh
            q = q + sh[:, :, None]
        t = np.asarray(rec['truth'], float)
        if p.shape != t.shape:
            continue
        e = t[:, :, None] - q
        pin = np.maximum(qlev * e, (qlev - 1) * e)
        s = dict(arm=rec['arm'], offset=rec['offset'], origin=rec['origin'], mae=float(np.abs(p - t).mean()), sql=float(2 * pin.mean() / sc), mase=float(np.abs(p - t).mean() / sc), bias=float((p - t).mean()))
        detail.append(s)
        per_arm.setdefault(rec['arm'], []).append(s)
    state['aux_detail'] = detail
    state['aux_diag'] = {k: {'n': len(v), 'sql': float(np.mean([x['sql'] for x in v])), 'mase': float(np.mean([x['mase'] for x in v])), 'bias': float(np.mean([x['bias'] for x in v]))} for k, v in per_arm.items()}
    if 'ctl' in per_arm:
        base = {x['offset']: x['sql'] for x in per_arm['ctl']}
        state['aux_winrate'] = {k: float(np.mean([x['sql'] < base.get(x['offset'], np.inf) for x in v])) for k, v in per_arm.items()}
    state['notes'].append('aux diag %s' % state['aux_diag'])
    for out, case in zip(outputs, cases):
        if case.get('auxiliary'):
            continue
        p = np.asarray(out['point'], float)
        q = np.asarray(out['quantiles'], float)
        sh = state.get('shifts', {}).get(('primary', CONFIG['primary_arm'], 0))
        if sh is not None and np.any(sh):
            p = p + sh
            q = q + sh[:, :, None]
        q = np.sort(q, axis=-1)
        p = np.clip(p, q[:, :, 0], q[:, :, 8])
        idx = list(case['target_indices'])
        point[idx] = p
        quant[idx] = q
        filled[idx] = True
    if not filled.all():
        miss = np.where(~filled)[0]
        last = state['hist'][miss][:, -H:]
        point[miss] = last
        quant[miss] = last[:, :, None] * np.ones(9)[None, None, :]
    return {'point': point, 'quantiles': quant, 'components': {'aux_diag': state['aux_diag']}}

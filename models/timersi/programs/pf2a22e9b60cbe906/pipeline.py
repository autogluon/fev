import numpy as np
from calendar_inputs import augment
import specialist
import tailcal
from corrections import Origin, apply_correction, scaled_pinball
AUX_OFFSETS = (13, 26, 39)
MIN_AUX_CONTEXT = 19
CAL_MIN_CONTEXT = 26
ADD_CALENDAR_INPUTS = True
TAIL_CALIBRATION = True
USE_SPECIALIST = True
LOG_VIEW = False
LOG_VIEW_MIN_CONTEXT = 26
BLEND = 0.5
MACRO_PAST = ('oil_price',)
NOOIL_MIN_CONTEXT = 26
SHORT_VIEW = None
SHORT_VIEW_MIN_GAIN = 6

def preprocess(view, card):
    hist = np.asarray(view['target_history'], float)
    obs = np.asarray(view['target_observed'], bool)
    L = int(view['cutoff_index'])
    H = int(view['horizon'])
    finite = hist[np.isfinite(hist)]
    return (view, {'N': hist.shape[0], 'H': H, 'L': L, 'hist': hist, 'obs': obs, 'ts': [str(t) for t in view['timestamps']], 'nonneg': bool(finite.size and np.all(finite >= 0))})

def engineer(prepared, card, state):
    state['origin'] = Origin(state['hist'], state['obs'], state['ts'], state['L'], state['H'], want_cal=state['L'] >= CAL_MIN_CONTEXT)
    kf0 = prepared['known_features']
    kn0 = list(prepared['known_names'])
    state['known_base'] = (np.asarray(kf0, float) if kf0 is not None and len(kf0) else None, kn0)
    kf, kn = augment(kf0, kn0, state['ts']) if ADD_CALENDAR_INPUTS else (kf0, kn0)
    state['known'] = (np.asarray(kf, float) if kf is not None and len(kf) else None, kn)
    return (prepared, state)

def _case(hist, po, kf, kn, names, lo, hi, H, N, **extra):
    c = {'target_indices': list(range(N)), 'targets': hist[:, lo:hi], 'past_only': None if po is None else po[:, lo:hi], 'known_future': None if kf is None else kf[:, lo:hi + H], 'past_names': names, 'known_names': kn}
    c.update(extra)
    return c

def select_context(variables, card, state):
    L, H, N = (state['L'], state['H'], state['N'])
    C = min(L, int(card['limits']['max_context']))
    po = np.asarray(variables['past_features'], float) if len(variables['past_features']) else None
    names = list(variables['past_names'])
    kf, kn = state['known']
    hist = variables['target_history']
    base_kf, base_kn = state['known_base']
    cases = [_case(hist, po, base_kf, base_kn, names, L - C, L, H, N, provenance='primary: official reference inputs, observed oil preserved')]
    meta = [{'kind': 'primary', 'end': int(L)}]
    if kf is not None and BLEND > 0:
        cases.append(_case(hist, po, kf, kn, names, L - C, L, H, N, alternative=True, provenance='alternative: same origin and context, plus four deterministic calendar known covariates'))
        meta.append({'kind': 'alt_calendar', 'end': int(L)})
    if SHORT_VIEW is not None and C - SHORT_VIEW >= SHORT_VIEW_MIN_GAIN:
        cases.append(_case(hist, po, base_kf, base_kn, names, L - SHORT_VIEW, L, H, N, alternative=True, provenance='alternative: most recent %d weeks only' % SHORT_VIEW))
        meta.append({'kind': 'alt_short', 'end': int(L)})
    keep = [j for j, n in enumerate(names) if n not in MACRO_PAST]
    if po is not None and len(keep) < len(names) and (L >= NOOIL_MIN_CONTEXT):
        cases.append(_case(hist, po[keep] if keep else None, base_kf, base_kn, [names[j] for j in keep], L - C, L, H, N, alternative=True, provenance='alternative: same origin and context, national oil_price left out'))
        meta.append({'kind': 'alt_nooil', 'end': int(L)})
    if LOG_VIEW and state['nonneg'] and (L >= LOG_VIEW_MIN_CONTEXT):
        lhist = np.log1p(np.maximum(hist, 0.0))
        cases.append(_case(lhist, po, base_kf, base_kn, names, L - C, L, H, N, alternative=True, provenance='alternative: same origin/context, log1p targets, reversed with expm1 in postprocess'))
        meta.append({'kind': 'alt_log', 'end': int(L)})
    for off in AUX_OFFSETS:
        end = L - off
        if off < H or end < MIN_AUX_CONTEXT or end + H > L:
            continue
        Ca = min(end, C)
        cases.append(_case(hist, po, kf, kn, names, end - Ca, end, H, N, auxiliary=True, origin_offset=int(off), provenance='auxiliary native backtest, origin -%dw, identical covariate composition' % off))
        meta.append({'kind': 'auxiliary', 'end': int(end), 'offset': int(off)})
    state['case_meta'] = meta
    return (cases, state)

def _diagnose(outputs, state):
    H = state['H']
    rep = []
    for i, m in enumerate(state['case_meta']):
        if i >= len(outputs) or m['kind'] != 'auxiliary' or outputs[i] is None:
            continue
        end = m['end']
        p = np.asarray(outputs[i]['point'], float).reshape(state['N'], H)
        q = np.asarray(outputs[i]['quantiles'], float).reshape(state['N'], H, 9)
        for d in range(state['N']):
            y = state['hist'][d][end:end + H]
            o = state['obs'][d][end:end + H]
            ctx = state['hist'][d][:end][state['obs'][d][:end]]
            ctx = ctx[np.isfinite(ctx)]
            if ctx.size < 2 or not np.all(o) or (not np.all(np.isfinite(y))):
                continue
            sc = float(np.mean(np.abs(np.diff(ctx))))
            if not np.isfinite(sc) or sc <= 1e-09:
                continue
            good = (y > 0) & (p[d] > 0)
            rep.append({'offset': m['offset'], 'end': end, 'scale': sc, 'sql_raw': scaled_pinball(y, q[d], sc), 'log_bias': float(np.mean(np.log(y[good] / p[d][good]))) if good.any() else None, 'log_bias_by_h': [float(np.log(y[t] / p[d][t])) if good[t] else None for t in range(H)], 'coverage': [float(np.mean(y <= q[d][:, j])) for j in range(9)]})
    return rep

def postprocess(outputs, cases, state):
    N, H = (state['N'], state['H'])
    meta = state['case_meta']
    point = np.asarray(outputs[0]['point'], float).reshape(N, H)
    quant = np.asarray(outputs[0]['quantiles'], float).reshape(N, H, 9)
    views, used = ([(point, quant)], ['primary'])
    for i, m in enumerate(meta):
        if m['kind'] not in ('alt_calendar', 'alt_nooil', 'alt_short', 'alt_log'):
            continue
        if i >= len(outputs) or outputs[i] is None:
            continue
        ap = np.asarray(outputs[i]['point'], float).reshape(N, H)
        aq = np.asarray(outputs[i]['quantiles'], float).reshape(N, H, 9)
        if m['kind'] == 'alt_log':
            ap = np.expm1(np.clip(ap, -1.0, 18.0))
            aq = np.expm1(np.clip(aq, -1.0, 18.0))
            aq = np.sort(aq, axis=2)
        if np.all(np.isfinite(ap)) and np.all(np.isfinite(aq)):
            views.append((ap, aq))
            used.append(m['kind'])
    point = np.mean([v[0] for v in views], axis=0)
    quant = np.mean([v[1] for v in views], axis=0)
    raw_p, raw_q = (point.copy(), quant.copy())
    org0 = state['origin']
    comp = {'L': int(state['L']), 'calendar_inputs': bool(ADD_CALENDAR_INPUTS), 'holiday_prior': {'fixed_mass': float(getattr(org0, 'fixed_mass', -1.0)), 'lam': float(getattr(org0, 'lam', -1.0)), 'prior_log': [round(float(v), 4) for v in getattr(org0, 'prior', [])], 'active': bool(float(np.max(getattr(org0, 'prior', np.zeros(1)))) > 1e-08)}, 'views': used}
    try:
        comp['replays'] = _diagnose(outputs, state)
    except Exception as exc:
        comp['replay_error'] = repr(exc)
    org = state['origin']
    new_p = np.empty_like(point)
    new_q = np.empty_like(quant)
    for d in range(N):
        new_p[d], new_q[d] = apply_correction(org, d, point[d], quant[d], 1.0, 1.0, 1.0, nonneg=state['nonneg'])
    if not (np.all(np.isfinite(new_p)) and np.all(np.isfinite(new_q))):
        return {'point': raw_p, 'quantiles': raw_q, 'components': comp}
    if TAIL_CALIBRATION:
        try:
            prior_vec = np.asarray(getattr(org, 'prior', np.zeros(H)), float)
            cold_event = float((1.0 - getattr(org, 'lam', 1.0)) * prior_vec.max())
            if cold_event > 0.02:
                m_lo, m_up, tinfo = (1.0, 1.0, {'skipped': 'cold fixed-event horizon', 'cold_event': round(cold_event, 4)})
                recs = []
            else:
                recs = tailcal.collect_replays(state, outputs, CAL_MIN_CONTEXT)
                m_lo, m_up, tinfo = tailcal.choose_multipliers(recs)
            protect = prior_vec > 1e-08
            if m_lo != 1.0 or m_up != 1.0:
                for d in range(N):
                    new_q[d] = tailcal.apply_to_final(new_q[d], m_lo, m_up, protect)
                if state['nonneg']:
                    new_q = np.maximum(new_q, 0.0)
            tinfo['protected_weeks'] = int(protect.sum())
            comp['tail_calibration'] = tinfo
        except Exception as exc:
            comp['tail_calibration'] = {'error': repr(exc)}
    if USE_SPECIALIST:
        try:
            new_p, new_q, sinfo = _specialist_blend(outputs, state, new_p, new_q)
            comp['specialist'] = sinfo
        except Exception as exc:
            comp['specialist'] = {'error': repr(exc)}
    return {'point': new_p, 'quantiles': new_q, 'components': comp}

def _specialist_blend(outputs, state, new_p, new_q):
    N, H, L = (state['N'], state['H'], state['L'])
    ts = state['ts']
    sinfo = {'fitted': False}
    org0 = state['origin']
    cold = float((1.0 - getattr(org0, 'lam', 1.0)) * np.max(getattr(org0, 'prior', np.zeros(1))))
    if cold > 0.02:
        sinfo['skipped'] = 'cold fixed-event horizon'
        return (new_p, new_q, sinfo)
    live = [specialist.fit_predict(state['hist'][d][:L], state['obs'][d][:L], ts[:L], ts[L:L + H]) for d in range(N)]
    if all((s is None for s in live)):
        sinfo['reason'] = 'no fit at live origin'
        return (new_p, new_q, sinfo)
    sinfo['fitted'] = True
    replays = []
    for i, m in enumerate(state['case_meta']):
        if m['kind'] != 'auxiliary' or i >= len(outputs) or outputs[i] is None:
            continue
        end = m['end']
        p = np.asarray(outputs[i]['point'], float).reshape(N, H)
        q = np.asarray(outputs[i]['quantiles'], float).reshape(N, H, 9)
        org_a = Origin(state['hist'], state['obs'], ts, end, H, want_cal=end >= CAL_MIN_CONTEXT)
        for d in range(N):
            y = state['hist'][d][end:end + H]
            o = state['obs'][d][end:end + H]
            ctx = state['hist'][d][:end][state['obs'][d][:end]]
            ctx = ctx[np.isfinite(ctx)]
            if ctx.size < 2 or not np.all(o) or (not np.all(np.isfinite(y))):
                continue
            sc = float(np.mean(np.abs(np.diff(ctx))))
            if not np.isfinite(sc) or sc <= 1e-09:
                continue
            sl = specialist.fit_predict(state['hist'][d][:end], state['obs'][d][:end], ts[:end], ts[end:end + H])
            if sl is None:
                continue
            pc, qc = apply_correction(org_a, d, p[d], q[d], 1.0, 1.0, 1.0, nonneg=state['nonneg'])
            replays.append((y, pc, qc, sl, sc))
    w, winfo = specialist.choose_weight(replays)
    sinfo.update(winfo)
    if w > 0:
        for d in range(N):
            if live[d] is None:
                continue
            new_p[d], new_q[d] = specialist.blend(new_p[d], new_q[d], live[d], w)
        if state['nonneg']:
            new_p = np.maximum(new_p, 0.0)
            new_q = np.maximum(new_q, 0.0)
    sinfo['applied_weight'] = w
    return (new_p, new_q, sinfo)

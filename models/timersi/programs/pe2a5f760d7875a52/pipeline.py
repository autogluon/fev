import numpy as np
from calendar_inputs import augment
from corrections import Origin, apply_correction, scaled_pinball
AUX_OFFSETS = (13, 26)
MIN_AUX_CONTEXT = 19
CAL_MIN_CONTEXT = 26
ADD_CALENDAR_INPUTS = True

def preprocess(view, card):
    hist = np.asarray(view['target_history'], float)
    obs = np.asarray(view['target_observed'], bool)
    L = int(view['cutoff_index'])
    H = int(view['horizon'])
    finite = hist[np.isfinite(hist)]
    return (view, {'N': hist.shape[0], 'H': H, 'L': L, 'hist': hist, 'obs': obs, 'ts': [str(t) for t in view['timestamps']], 'nonneg': bool(finite.size and np.all(finite >= 0))})

def engineer(prepared, card, state):
    state['origin'] = Origin(state['hist'], state['obs'], state['ts'], state['L'], state['H'], want_cal=state['L'] >= CAL_MIN_CONTEXT)
    kf = prepared['known_features']
    kn = list(prepared['known_names'])
    if ADD_CALENDAR_INPUTS:
        kf, kn = augment(kf, kn, state['ts'])
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
    cases = [_case(hist, po, kf, kn, names, L - C, L, H, N, provenance='primary: reference inputs + 4 deterministic calendar known covariates; observed oil preserved')]
    meta = [{'kind': 'primary', 'end': int(L)}]
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
    point = np.asarray(outputs[0]['point'], float).reshape(N, H)
    quant = np.asarray(outputs[0]['quantiles'], float).reshape(N, H, 9)
    raw_p, raw_q = (point.copy(), quant.copy())
    comp = {'L': int(state['L']), 'calendar_inputs': bool(ADD_CALENDAR_INPUTS)}
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
    return {'point': new_p, 'quantiles': new_q, 'components': comp}

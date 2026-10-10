import os
import sys
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import cumlib
MAX_CTX_DEFAULT = 15360
ONSET_MULT, ONSET_SIG, ONSET_KSHAPE = (600.0, 1.2, 3.2)
BAND_BETA = 0.93
MIN_LONG_HISTORY = 12

def preprocess(view, card):
    hist = np.asarray(view['target_history'], dtype=float)
    obs = np.asarray(view['target_observed'], dtype=bool)
    D, L = hist.shape
    clean = np.empty_like(hist)
    profiles = []
    for d in range(D):
        y, _ = cumlib.clean_history(hist[d], obs[d])
        clean[d] = y
        profiles.append(cumlib.cumulative_profile(y))
    prepared = dict(view)
    prepared['target_history'] = clean
    state = {'N': D, 'H': int(view['horizon']), 'L': int(L), 'profiles': profiles, 'last': clean[:, -1].copy() if L else np.zeros(D), 'item_id': view.get('item_id')}
    return (prepared, state)

def engineer(prepared, card, state):
    hist = prepared['target_history']
    D, L = hist.shape
    inc = np.zeros_like(hist)
    if L >= 2:
        inc[:, 1:] = np.diff(hist, axis=1)
    inc[:, 0] = 0.0
    for d, p in enumerate(state['profiles']):
        if p['is_cumulative']:
            inc[d] = np.maximum(inc[d], 0.0)
    variables = dict(prepared)
    variables['increments'] = inc
    state['inc_stats'] = [{'mean': float(inc[d, 1:].mean()) if L > 1 else 0.0, 'last': float(inc[d, -1]) if L else 0.0, 'nz': int((inc[d, 1:] > 0).sum()) if L > 1 else 0} for d in range(D)]
    state['onset_w'] = [cumlib.onset_weight(state['profiles'][d], inc[d, 1:]) for d in range(D)]
    state['inc_past'] = [inc[d, 1:] for d in range(D)]
    state['rate_sd'] = [float('nan')] * D
    return (variables, state)

def select_context(variables, card, state):
    L = int(variables['cutoff_index'])
    H = int(variables['horizon'])
    max_ctx = int(card.get('limits', {}).get('max_context', MAX_CTX_DEFAULT))
    use_inc = [p['is_cumulative'] and L >= 3 for p in state['profiles']]
    state['use_inc'] = use_inc
    inc = variables['increments']
    hist = variables['target_history']
    rows, idx, mode = ([], [], [])
    for d in range(state['N']):
        if use_inc[d]:
            rows.append(inc[d, 1:])
        else:
            rows.append(hist[d])
        idx.append(d)
        mode.append('increment' if use_inc[d] else 'level')
    C = min(min((len(r) for r in rows)), max_ctx)
    C = max(C, 1)
    targets = np.stack([r[-C:] for r in rows], axis=0)
    po = np.asarray(variables['past_features'], dtype=float) if len(variables['past_features']) else None
    pf = np.asarray(variables['known_features'], dtype=float) if len(variables['known_features']) else None
    state['C'] = int(C)
    return ([{'target_indices': idx, 'targets': targets, 'past_only': po[:, -C:] if po is not None and po.size else None, 'known_future': pf[:, -(C + H):] if pf is not None and pf.size else None, 'past_names': variables['past_names'], 'known_names': variables['known_names'], 'provenance': 'increment-domain target (%s), C=%d, reintegrated in postprocess' % ('/'.join(mode), C)}], state)

def postprocess(outputs, cases, state):
    D, H = (state['N'], state['H'])
    point = np.empty((D, H))
    quant = np.empty((D, H, 9))
    for output, case in zip(outputs, cases):
        op = np.asarray(output['point'], float)
        oq = np.asarray(output['quantiles'], float)
        oq = cumlib.enforce_quantile_sort(oq)
        for j, d in enumerate(case['target_indices']):
            y_last = float(state['last'][d])
            if state['use_inc'][d]:
                new_p = np.maximum(op[j], 0.0)
                new_q = np.maximum(oq[j], 0.0)
                cp, cq = cumlib.integrate(new_p, new_q, y_last, width_beta=BAND_BETA, lam=1.0)
                if state['L'] >= MIN_LONG_HISTORY:
                    v = cumlib.log_rate_innovation_sd(state['inc_past'][d])
                    state['rate_sd'][d] = v if v is not None else float('nan')
                    inc_q = np.maximum(cq - y_last, 0.0)
                    inc_q = cumlib.widen_increment_band(inc_q, v)
                    cq = y_last + inc_q
            else:
                cp, cq = (op[j], oq[j])
            w = state['onset_w'][d]
            if w > 0.0:
                prior = cumlib.onset_fan(y_last, H, med_mult=ONSET_MULT, sig=ONSET_SIG, k_shape=ONSET_KSHAPE)
                blended = cumlib.log_blend(cq, prior, w)
                cq = np.maximum(blended, cq)
                cp = cq[:, 4]
            if state['profiles'][d]['is_cumulative']:
                cp, cq = cumlib.project_cumulative(cp, cq, y_last)
            point[d] = cp
            quant[d] = cq
    return {'point': point, 'quantiles': quant, 'components': {'mode': ['increment' if u else 'level' for u in state['use_inc']], 'onset_weight': [float(x) for x in state['onset_w']], 'own_past_rate_sd': [float(x) for x in state['rate_sd']]}}

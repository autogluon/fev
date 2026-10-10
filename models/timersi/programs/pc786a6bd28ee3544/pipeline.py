import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import grouping
import regime
import robust
MODE = 'fixed_context'
BUCKETS = (672, 2016)
FIXED_CONTEXT = 672
MAX_GROUP = 32

def preprocess(view, card):
    H = robust.nan_to_finite(np.asarray(view['target_history'], dtype=np.float64))
    state = {'N': H.shape[0], 'H': int(view['horizon']), 'L': int(view['cutoff_index']), 'hist': H, 'season': int(card.get('seasonality') or 24)}
    state['sscale'] = robust.seasonal_scale(H, state['season'])
    prepared = dict(view)
    prepared['target_history'] = H
    return (prepared, state)

def engineer(prepared, card, state):
    return (prepared, state)

def _slices(hist, L, card, state):
    Cmax = min(L, int(card['limits']['max_context']))
    if MODE == 'univariate':
        return ([(Cmax, [i]) for i in range(state['N'])], 'One native request per target; full context')
    if MODE == 'correlation':
        return ([(Cmax, g) for g in grouping.correlation_blocks(hist, 26)], 'Rank-correlation blocks; full context')
    if MODE == 'fixed_context':
        C = min(Cmax, FIXED_CONTEXT)
        return ([(C, list(range(state['N'])))], 'Reference single 52-variate block, context capped at %d hours' % C)
    if MODE != 'regime_context':
        return ([(Cmax, list(range(state['N'])))], 'Reference single block, full context')
    chosen = regime.context_buckets(hist, Cmax, levels=BUCKETS)
    out = []
    for c in sorted(set(chosen.tolist()), reverse=True):
        idx = [int(i) for i in np.where(chosen == c)[0]]
        if len(idx) > MAX_GROUP and c != Cmax:
            for g in grouping.correlation_blocks(hist[idx], MAX_GROUP):
                out.append((int(c), [idx[j] for j in g]))
        else:
            out.append((int(c), idx))
    return (out, 'Regime-adaptive context buckets %s from each origin own past' % (BUCKETS,))

def select_context(variables, card, state):
    H = variables['horizon']
    hist = state['hist']
    po = np.asarray(variables['past_features'], dtype=np.float64) if len(variables['past_features']) else None
    pf = np.asarray(variables['known_features'], dtype=np.float64) if len(variables['known_features']) else None
    plan, prov = _slices(hist, state['L'], card, state)
    flat = sorted((i for _, g in plan for i in g))
    assert flat == list(range(state['N'])), 'every target exactly once'
    cases = []
    for C, g in plan:
        C = int(max(1, min(C, state['L'])))
        cases.append({'target_indices': list(g), 'targets': hist[g][:, -C:], 'past_only': po[:, -C:] if po is not None else None, 'known_future': pf[:, -(C + H):] if pf is not None else None, 'past_names': variables['past_names'], 'known_names': variables['known_names'], 'provenance': '%s | C=%d, %d targets' % (prov, C, len(g))})
    state['plan'] = [(int(C), list(map(int, g))) for C, g in plan]
    return (cases, state)

def postprocess(outputs, cases, state):
    N, H = (state['N'], state['H'])
    point = np.empty((N, H), dtype=np.float64)
    quantiles = np.empty((N, H, 9), dtype=np.float64)
    for output, case in zip(outputs, cases):
        idx = case['target_indices']
        point[idx] = np.asarray(output['point'], dtype=np.float64)
        quantiles[idx] = np.asarray(output['quantiles'], dtype=np.float64)
    quantiles = np.sort(quantiles, axis=2)
    point = np.where(np.isfinite(point), point, quantiles[:, :, 4])
    return {'point': point, 'quantiles': quantiles}

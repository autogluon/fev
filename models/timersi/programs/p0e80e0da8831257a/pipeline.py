import numpy as np
USE_PAST_COVARIATES = False
try:
    from util_targets import degenerate_mask
    from util_damping import damping_lambda
except Exception:
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from util_targets import degenerate_mask
    from util_damping import damping_lambda

def preprocess(view, card):
    hist = np.asarray(view['target_history'], float)
    obs = np.asarray(view['target_observed'], bool)
    dead = degenerate_mask(hist, obs)
    H = int(view['horizon'])
    anchors = np.zeros(hist.shape[0])
    lams = np.ones(hist.shape[0])
    betas = [None] * hist.shape[0]
    for d in range(hist.shape[0]):
        m = obs[d] & np.isfinite(hist[d])
        if m.sum() == 0:
            continue
        y = hist[d][m]
        anchors[d] = y[-1]
        lam, beta = damping_lambda(y, horizon=H)
        lams[d] = lam
        betas[d] = beta
    state = {'N': len(view['target_ids']), 'H': H, 'dead': dead, 'live': [i for i in range(len(dead)) if not dead[i]], 'names': list(view['target_ids']), 'anchor': anchors, 'lam': lams, 'beta': betas}
    return (view, state)

def engineer(prepared, card, state):
    return (prepared, state)

def select_context(variables, card, state):
    C = min(variables['cutoff_index'], card['limits']['max_context'])
    H = variables['horizon']
    po = np.asarray(variables['past_features'], float)
    pf = np.asarray(variables['known_features'], float)
    hist = np.asarray(variables['target_history'], float)
    live = state['live']
    dead = [i for i in range(state['N']) if i not in set(live)]
    if not live:
        live, dead = (list(range(state['N'])), [])
        state['live'], state['dead'] = (live, [False] * state['N'])
    cases = [{'target_indices': list(live), 'targets': hist[live][:, -C:], 'past_only': po[:, -C:] if po.size and USE_PAST_COVARIATES else None, 'known_future': pf[:, -(C + H):] if pf.size else None, 'past_names': variables['past_names'] if USE_PAST_COVARIATES else [], 'known_names': variables['known_names'], 'provenance': 'observed targets only, max context' if not USE_PAST_COVARIATES else 'observed targets + official global past covariates, max context'}]
    if dead:
        cases.append({'target_indices': list(dead), 'targets': hist[dead][:, -C:], 'past_only': None, 'known_future': None, 'past_names': [], 'known_names': [], 'provenance': 'never-observed placeholder channels isolated (unscored by FEV)'})
    return (cases, state)

def postprocess(outputs, cases, state):
    N, H = (state['N'], state['H'])
    point = np.zeros((N, H))
    quantiles = np.zeros((N, H, 9))
    for output, case in zip(outputs, cases):
        idx = list(case['target_indices'])
        point[idx] = np.asarray(output['point'], float)
        quantiles[idx] = np.asarray(output['quantiles'], float)
    anchor = np.asarray(state['anchor'], float)[:, None]
    lam = np.asarray(state['lam'], float)[:, None]
    damped = anchor + lam * (point - anchor)
    shift = damped - point
    point = damped
    quantiles = quantiles + shift[:, :, None]
    point = np.nan_to_num(point, nan=0.0, posinf=0.0, neginf=0.0)
    quantiles = np.nan_to_num(quantiles, nan=0.0, posinf=0.0, neginf=0.0)
    quantiles = np.sort(quantiles, axis=-1)
    return {'point': point, 'quantiles': quantiles}

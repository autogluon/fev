import numpy as np
from context import choose_context
from calibrate import spread_factor, rescale

def preprocess(view, card):
    N = len(view['target_ids'])
    H = view['horizon']
    hist = np.array(view['target_history'], dtype=float)
    for d in range(N):
        row = hist[d]
        bad = ~np.isfinite(row)
        if bad.any():
            good = np.where(~bad)[0]
            row[bad] = np.interp(np.where(bad)[0], good, row[good]) if good.size else 0.0
    prepared = dict(view)
    prepared['target_history'] = hist
    return (prepared, {'N': N, 'H': H, 'history': hist})

def engineer(prepared, card, state):
    return (prepared, state)

def select_context(variables, card, state):
    H = variables['horizon']
    po = variables['past_features']
    pf = variables['known_features']
    hist = np.asarray(variables['target_history'], dtype=float)
    C = choose_context(variables['cutoff_index'], card['limits']['max_context'])
    state['context'] = C
    return ([{'target_indices': list(range(state['N'])), 'targets': hist[:, -C:], 'past_only': po[:, -C:] if len(po) else None, 'known_future': pf[:, -(C + H):] if len(pf) else None, 'past_names': variables['past_names'], 'known_names': variables['known_names'], 'provenance': f'raw official levels, context={C} months (two business cycles)'}], state)

def postprocess(outputs, cases, state):
    N, H = (state['N'], state['H'])
    hist = state['history']
    point = np.empty((N, H))
    quantiles = np.empty((N, H, 9))
    for output, case in zip(outputs, cases):
        p = np.asarray(output['point'], dtype=float)
        q = np.sort(np.asarray(output['quantiles'], dtype=float), axis=-1)
        for j, d in enumerate(case['target_indices']):
            f = spread_factor(hist[d], q[j])
            quantiles[d] = rescale(q[j], f)
            point[d] = p[j]
    point = np.where(np.isfinite(point), point, 0.0)
    quantiles = np.where(np.isfinite(quantiles), quantiles, 0.0)
    return {'point': point, 'quantiles': quantiles}

import numpy as np
from imputation import reconstruct
HALFLIFE_DAYS = 56
RECON_ITERS = 6

def preprocess(view, card):
    prepared = dict(view)
    th = np.asarray(view['target_history'], float)
    ob = np.asarray(view['target_observed'], bool)
    L = int(view['cutoff_index'])
    state = {'N': len(view['target_ids']), 'H': int(view['horizon']), 'names': list(view['target_names']), 'L': L, 'missing_fraction': float(1.0 - ob.mean()), 'trailing_gap': [int(len(r) - 1 - np.where(r)[0][-1]) if r.any() else len(r) for r in ob]}
    state['clean_history'] = th
    if ob.all():
        state['reconstructed'] = False
        return (prepared, state)
    kf = np.asarray(view['known_features'], float)
    try:
        filled = reconstruct(th, ob, np.asarray(view['timestamps']), kf[:, :L] if kf.size else None, halflife_h=24 * HALFLIFE_DAYS, iters=RECON_ITERS)
        if np.isfinite(filled).all() and (filled > 0).all():
            prepared['target_history'] = filled
            state['clean_history'] = filled
            state['reconstructed'] = True
        else:
            state['reconstructed'] = False
    except Exception as exc:
        state['reconstructed'] = False
        state['reconstruction_error'] = repr(exc)
    return (prepared, state)

def engineer(prepared, card, state):
    try:
        th = np.asarray(prepared['target_history'], float)
        ob = np.asarray(prepared['target_observed'], bool)
        tail = th[:, -336:]
        state['tail_cv'] = [float(np.std(np.log(np.clip(r, 0.001, None)))) for r in tail]
        state['obs_rate'] = [float(r.mean()) for r in ob]
    except Exception as exc:
        state['engineer_error'] = repr(exc)
    return (prepared, state)

def select_context(variables, card, state):
    L = int(variables['cutoff_index'])
    C = int(min(L, card['limits']['max_context']))
    H = int(variables['horizon'])
    po = np.asarray(variables['past_features'], float)
    pf = np.asarray(variables['known_features'], float)
    return ([{'target_indices': list(range(state['N'])), 'targets': np.asarray(variables['target_history'], float)[:, L - C:L], 'past_only': po[:, L - C:L] if po.size else None, 'known_future': pf[:, L - C:L + H] if pf.size else None, 'past_names': list(variables['past_names']), 'known_names': list(variables['known_names']), 'provenance': 'Causally reconstructed target history (cross-pollutant + known meteorology + calendar), full reference context %d h' % C}], state)

def postprocess(outputs, cases, state):
    N, H = (state['N'], state['H'])
    point = np.empty((N, H))
    quantiles = np.empty((N, H, 9))
    for output, case in zip(outputs, cases):
        if case.get('auxiliary'):
            continue
        point[case['target_indices']] = output['point']
        quantiles[case['target_indices']] = output['quantiles']
    quantiles = np.sort(quantiles, axis=2)
    return {'point': point, 'quantiles': quantiles}

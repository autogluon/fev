import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from specialist import day_grid, build_features, residual_quantiles, constructed_known
from lear import LEAR
LEVELS = np.arange(1, 10) / 10.0
WINDOWS = (56, 84, 364, 1092, 1456)
W_NATIVE = 0.8
ADD_KNOWN = True

def preprocess(view, card):
    state = {'N': len(view['target_ids']), 'H': view['horizon'], 'spec': None, 'err': None}
    try:
        g = day_grid(view, card)
        if g is not None:
            state['grid'] = g
    except Exception as exc:
        state['err'] = 'grid:%s' % exc
    return (view, state)

def engineer(prepared, card, state):
    g = state.get('grid')
    if ADD_KNOWN and g is not None:
        try:
            kf, kn = constructed_known(prepared, g)
            prepared['known_features'] = kf
            prepared['known_names'] = kn
            state['known_added'] = len(kn)
        except Exception as exc:
            state['err'] = 'known:%s' % exc
    if g is None:
        return (prepared, state)
    try:
        F = build_features(g)
        P = g['P']
        D = g['nd']
        mdl = LEAR(criterion='aic')
        point = mdl.fit_predict(F, P, D, windows=WINDOWS)
        qz, s = residual_quantiles(F, P, D, LEVELS)
        state['spec'] = {'point': np.asarray(point, float), 'qz': qz, 's': s}
    except Exception as exc:
        state['err'] = 'fit:%s' % exc
        state['spec'] = None
    return (prepared, state)

def select_context(variables, card, state):
    C = min(variables['cutoff_index'], card['limits']['max_context'])
    H = variables['horizon']
    po = variables['past_features']
    pf = variables['known_features']
    return ([{'target_indices': list(range(state['N'])), 'targets': variables['target_history'][:, -C:], 'past_only': po[:, -C:] if len(po) else None, 'known_future': pf[:, -(C + H):] if len(pf) else None, 'past_names': variables['past_names'], 'known_names': variables['known_names'], 'provenance': 'Official reference inputs, all targets/covariates, max_context=15360'}], state)

def postprocess(outputs, cases, state):
    N, H = (state['N'], state['H'])
    point = np.empty((N, H))
    quantiles = np.empty((N, H, 9))
    for output, case in zip(outputs, cases):
        point[case['target_indices']] = output['point']
        quantiles[case['target_indices']] = output['quantiles']
    components = {'native_point': point.copy(), 'native_quantiles': quantiles.copy(), 'note': state.get('err')}
    spec = state.get('spec')
    if spec is not None and spec['qz'] is not None and (N == 1):
        lp = spec['point']
        lq = lp[:, None] + spec['s'][:, None] * spec['qz'][None, :]
        lq = np.sort(lq, axis=1)
        components['lear_point'] = lp.copy()
        components['lear_quantiles'] = lq.copy()
        w = W_NATIVE
        q = w * quantiles[0] + (1.0 - w) * lq
        q = np.sort(q, axis=1)
        quantiles[0] = q
        point[0] = q[:, 4]
    return {'point': point, 'quantiles': quantiles, 'components': components}

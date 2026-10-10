import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from . import transforms as tf
except ImportError:
    import transforms as tf

def preprocess(view, card):
    hist = np.asarray(view['target_history'], dtype=float)
    past = np.asarray(view['past_features'], dtype=float) if len(view['past_features']) else np.zeros((0, 0))
    known = np.asarray(view['known_features'], dtype=float) if len(view['known_features']) else np.zeros((0, 0))
    target_kinds = [tf.choose_transform(hist[i], view['target_names'][i]) for i in range(hist.shape[0])]
    past_kinds = [tf.choose_transform(past[i], view['past_names'][i], min_snr=0.0, min_frac_up=0.0) for i in range(past.shape[0])]
    known_kinds = ['identity'] * (known.shape[0] if known.size else 0)
    prepared = dict(view)
    prepared['target_history'] = tf.transform_matrix(hist, target_kinds) if hist.size else hist
    if past.size:
        prepared['past_features'] = tf.transform_matrix(past, past_kinds)
    if known.size:
        prepared['known_features'] = tf.transform_matrix(known, known_kinds)
    state = {'N': hist.shape[0], 'H': view['horizon'], 'target_kinds': target_kinds, 'past_kinds': past_kinds, 'known_kinds': known_kinds, 'target_names': list(view['target_names'])}
    return (prepared, state)

def engineer(prepared, card, state):
    return (prepared, state)

def select_context(variables, card, state):
    C = min(variables['cutoff_index'], card['limits']['max_context'])
    H = variables['horizon']
    po = variables['past_features']
    pf = variables['known_features']
    case = {'target_indices': list(range(state['N'])), 'targets': np.asarray(variables['target_history'])[:, -C:], 'past_only': np.asarray(po)[:, -C:] if len(po) else None, 'known_future': np.asarray(pf)[:, -(C + H):] if len(pf) else None, 'past_names': variables['past_names'], 'known_names': variables['known_names'], 'provenance': 'All targets/covariates, max context, origin-local log normalisation: ' + ','.join((f'{n}:{k}' for n, k in zip(state['target_names'], state['target_kinds'])))}
    return ([case], state)

def postprocess(outputs, cases, state):
    N, H = (state['N'], state['H'])
    point = np.empty((N, H))
    quantiles = np.empty((N, H, 9))
    for output, case in zip(outputs, cases):
        idx = list(case['target_indices'])
        p = np.asarray(output['point'], dtype=float)
        q = np.asarray(output['quantiles'], dtype=float)
        for r, t in enumerate(idx):
            kind = state['target_kinds'][t]
            point[t] = tf.inverse(p[r], kind)
            quantiles[t] = tf.inverse(q[r], kind)
    quantiles = np.sort(quantiles, axis=-1)
    return {'point': point, 'quantiles': quantiles}

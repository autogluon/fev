import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import common
CONTEXT = 2880

def preprocess(view, card):
    return (view, {'N': len(view['target_ids']), 'H': view['horizon']})

def engineer(prepared, card, state):
    return (prepared, state)

def select_context(variables, card, state):
    L = variables['cutoff_index']
    cap = min(L, card['limits']['max_context'])
    C = int(max(1, min(cap, CONTEXT)))
    H = variables['horizon']
    po = variables['past_features']
    pf = variables['known_features']
    state['C'] = C
    return ([{'target_indices': list(range(state['N'])), 'targets': variables['target_history'][:, -C:], 'past_only': po[:, -C:] if len(po) else None, 'known_future': pf[:, -(C + H):] if len(pf) else None, 'past_names': variables['past_names'], 'known_names': variables['known_names'], 'provenance': 'All targets jointly, trailing context=%d (30d) instead of 15360' % C}], state)

def postprocess(outputs, cases, state):
    point = np.empty((state['N'], state['H']))
    quantiles = np.empty((state['N'], state['H'], 9))
    for output, case in zip(outputs, cases):
        point[case['target_indices']] = output['point']
        quantiles[case['target_indices']] = output['quantiles']
    return {'point': point, 'quantiles': quantiles}

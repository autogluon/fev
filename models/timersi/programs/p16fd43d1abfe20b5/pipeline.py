import numpy as np
from operating_state import split as _orders_split

def preprocess(view, card):
    past_orders, future_orders = _orders_split(view)
    state = {'N': len(view['target_ids']), 'H': view['horizon'], 'L': int(view['cutoff_index']), 'past_orders': past_orders, 'future_orders': future_orders}
    return (view, state)

def engineer(prepared, card, state):
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
    fut = state.get('future_orders')
    if fut is not None and fut.shape[0] == H:
        closed = fut <= 0.0
        if closed.any():
            point[:, closed] = 0.0
            quantiles[:, closed, :] = 0.0
    point = np.nan_to_num(point, nan=0.0, posinf=0.0, neginf=0.0)
    quantiles = np.nan_to_num(quantiles, nan=0.0, posinf=0.0, neginf=0.0)
    quantiles = np.sort(quantiles, axis=-1)
    return {'point': point, 'quantiles': quantiles}

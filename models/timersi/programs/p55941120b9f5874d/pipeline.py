import numpy as np
from calib import calibrate_spread

def preprocess(view, card):
    state = {'N': len(view['target_ids']), 'H': view['horizon'], 'history': np.asarray(view['target_history'], dtype=np.float64)}
    return (view, state)

def engineer(prepared, card, state):
    return (prepared, state)

def select_context(variables, card, state):
    C = min(variables['cutoff_index'], card['limits']['max_context'])
    H = variables['horizon']
    po = variables['past_features']
    pf = variables['known_features']
    return ([{'target_indices': list(range(state['N'])), 'targets': variables['target_history'][:, -C:], 'past_only': po[:, -C:] if len(po) else None, 'known_future': pf[:, -(C + H):] if len(pf) else None, 'past_names': variables['past_names'], 'known_names': variables['known_names'], 'provenance': 'Reference inputs (probes 2 and 4 showed every input-side change is harmful); quantiles recalibrated in postprocess'}], state)

def postprocess(outputs, cases, state):
    N, H = (state['N'], state['H'])
    point = np.empty((N, H))
    quantiles = np.empty((N, H, 9))
    for output, case in zip(outputs, cases):
        point[case['target_indices']] = output['point']
        quantiles[case['target_indices']] = output['quantiles']
    try:
        quantiles, _ = calibrate_spread(quantiles, state['history'], H, weight=0.4, floor=0.2, ceiling=1.0, halflives=(6.0, 24.0, 96.0), stride=12, max_origins=600, n_blocks=8, ref_p=0.2, lower_extra=0.92)
    except Exception:
        pass
    point = quantiles[:, :, 4]
    return {'point': point, 'quantiles': quantiles}

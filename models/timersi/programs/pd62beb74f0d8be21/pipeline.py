import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import solar
FLOOR_FRAC = 0.03

def preprocess(view, card):
    state = {'N': len(view['target_ids']), 'H': int(view['horizon']), 'L': int(view['cutoff_index'])}
    try:
        s = solar.solar_structure(view)
        csf = np.maximum(s['cs'], FLOOR_FRAC * s['day_peak'])
        csf = np.where(np.isfinite(csf) & (csf > 0), csf, 1.0)
        s['csf'] = csf
        state['solar'] = s
    except Exception as exc:
        state['solar'] = None
        state['solar_error'] = repr(exc)
    return (view, state)

def engineer(prepared, card, state):
    s = state.get('solar')
    if s is None:
        return (prepared, state)
    view = dict(prepared)
    L = state['L']
    y = np.asarray(view['target_history'], dtype=float)
    z = y / s['csf'][None, :L]
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    z = np.clip(z, 0.0, 3.0)
    view['target_history'] = z
    state['normalised'] = True
    extra = np.vstack([s['cs'], 100.0 * s['elev']])
    names = np.array(['cs_envelope', 'solar_elev'])
    kf = np.asarray(view['known_features'], dtype=float)
    if kf.size:
        view['known_features'] = np.vstack([kf, extra])
        view['known_names'] = np.concatenate([np.asarray(view['known_names']), names])
    else:
        view['known_features'], view['known_names'] = (extra, names)
    view['known_features'] = np.nan_to_num(view['known_features'], nan=0.0, posinf=0.0, neginf=0.0)
    return (view, state)

def select_context(variables, card, state):
    C = min(int(variables['cutoff_index']), int(card['limits']['max_context']))
    H = int(variables['horizon'])
    po = np.asarray(variables['past_features'])
    pf = np.asarray(variables['known_features'])
    return ([{'target_indices': list(range(state['N'])), 'targets': np.asarray(variables['target_history'])[:, -C:], 'past_only': po[:, -C:] if po.size else None, 'known_future': pf[:, -(C + H):] if pf.size else None, 'past_names': variables['past_names'], 'known_names': variables['known_names'], 'provenance': 'Clear-sky-index normalised target + reference covariates + past-derived clear-sky envelope / solar elevation'}], state)

def postprocess(outputs, cases, state):
    N, H, L = (state['N'], state['H'], state['L'])
    point = np.empty((N, H))
    quantiles = np.empty((N, H, 9))
    for output, case in zip(outputs, cases):
        point[case['target_indices']] = output['point']
        quantiles[case['target_indices']] = output['quantiles']
    s = state.get('solar')
    if s is not None and state.get('normalised'):
        csf = s['csf'][L:L + H]
        point = point * csf[None, :]
        quantiles = quantiles * csf[None, :, None]
    if s is not None:
        dark = np.asarray(s['dark'])[L:L + H]
        if dark.shape[0] == H:
            point[:, dark] = 0.0
            quantiles[:, dark, :] = 0.0
    point = np.maximum(np.nan_to_num(point, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    quantiles = np.maximum(np.nan_to_num(quantiles, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    return {'point': point, 'quantiles': np.sort(quantiles, axis=2)}

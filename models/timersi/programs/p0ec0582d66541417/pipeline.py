import os
import sys
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from calendar_tools import steps_per_year
from calibrate import mase_scale, recentre, seasonal_volatility_ratio, shrink
from seasonal_index import build_known_future

def _clean_history(view):
    hist = np.array(view['target_history'], dtype=float)
    obs = np.array(view['target_observed'], dtype=bool) if len(view.get('target_observed', [])) else None
    for d in range(hist.shape[0]):
        row = hist[d]
        bad = ~np.isfinite(row)
        if obs is not None and obs.shape == hist.shape:
            bad = bad | ~obs[d]
        if bad.all():
            row[:] = 0.0
        elif bad.any():
            idx = np.where(~bad)[0]
            row[bad] = np.interp(np.where(bad)[0], idx, row[idx])
        hist[d] = row
    return hist

def preprocess(view, card):
    hist = _clean_history(view)
    prepared = dict(view)
    prepared['target_history'] = hist
    state = {'N': hist.shape[0], 'H': int(view['horizon']), 'L': int(view['cutoff_index']), 'item': view.get('item_id'), 'history': hist, 'stamps': list(view['timestamps'])}
    return (prepared, state)

def engineer(prepared, card, state):
    L, H = (state['L'], state['H'])
    stamps = state['stamps']
    past_stamps = stamps[:L]
    future_stamps = stamps[L:L + H]
    spy = steps_per_year(past_stamps) if len(past_stamps) >= 3 else None
    state['scale'] = mase_scale(state['history'])
    try:
        state['vol_ratio'] = seasonal_volatility_ratio(state['history'], past_stamps, future_stamps, spy)
    except Exception:
        state['vol_ratio'] = np.ones(state['N'])
    state['steps_per_year'] = spy
    try:
        extra, extra_names = build_known_future(state['history'], stamps, L, spy)
    except Exception:
        extra, extra_names = (np.zeros((0, len(stamps))), [])
    if extra.shape[0] and (not np.all(np.isfinite(extra))):
        extra, extra_names = (np.zeros((0, len(stamps))), [])
    state['extra_known'] = extra
    state['extra_known_names'] = extra_names
    return (prepared, state)

def select_context(variables, card, state):
    cap = int(card.get('limits', {}).get('max_context', 15360))
    C = max(1, min(state['L'], cap))
    H = state['H']
    po = np.asarray(variables['past_features'])
    pf = np.asarray(variables['known_features'])
    extra = state['extra_known']
    names = list(variables['known_names'])
    if extra.shape[0]:
        window = extra[:, state['L'] - C:state['L'] + H]
        pf = np.vstack([pf[:, -(C + H):], window]) if pf.size else window
        names = names + list(state['extra_known_names'])
    case = {'target_indices': list(range(state['N'])), 'targets': state['history'][:, -C:], 'past_only': po[:, -C:] if po.size else None, 'known_future': pf if pf.size else None, 'past_names': variables['past_names'], 'known_names': names, 'provenance': 'Reference targets, context=%d, constructed known-future calendar channels: %s' % (C, ','.join(names) or 'none')}
    return ([case], state)

def postprocess(outputs, cases, state):
    N, H = (state['N'], state['H'])
    point = np.empty((N, H))
    quant = np.empty((N, H, 9))
    for output, case in zip(outputs, cases):
        point[case['target_indices']] = np.asarray(output['point'], dtype=float)
        quant[case['target_indices']] = np.asarray(output['quantiles'], dtype=float)
    quant = np.sort(quant, axis=-1)
    quant = shrink(quant, state['vol_ratio'])
    point, quant = recentre(quant, state['scale'])
    comp = {'vol_ratio': np.asarray(state['vol_ratio'], dtype=float).tolist(), 'mase_scale': np.asarray(state['scale'], dtype=float).tolist()}
    return {'point': point, 'quantiles': quant, 'components': comp}

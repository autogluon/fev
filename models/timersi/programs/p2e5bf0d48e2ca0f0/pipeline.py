import numpy as np
from calibrate import drift_shrink_shift, envelope_guard, tail_widen
from robust import seasonal_scale
DRIFT_GAMMA = 0.6
RAMP_START, RAMP_END = (1, 14)
LEVEL_WINDOW = 9
ANCHOR_DAYS = 7
ENV_BASE, ENV_GROWTH, ENV_STRENGTH = (1.5, 1.0, 0.6)
TAIL_WIDEN, TAIL_START = (0.12, 14)

def preprocess(view, card):
    hist = np.asarray(view['target_history'], dtype=float)
    obs = np.asarray(view['target_observed'])
    if not np.all(np.isfinite(hist)) or not np.all(obs):
        for d in range(hist.shape[0]):
            row = hist[d]
            bad = ~np.isfinite(row) | ~np.asarray(obs[d], bool)
            if bad.all():
                row[:] = 0.0
            elif bad.any():
                good = np.where(~bad)[0]
                row[bad] = np.interp(np.where(bad)[0], good, row[good])
    prepared = dict(view)
    prepared['target_history'] = hist
    state = {'N': hist.shape[0], 'H': int(view['horizon']), 'L': int(view['cutoff_index']), 'item_id': view.get('item_id'), 'history': hist, 'names': list(view['target_names'])}
    return (prepared, state)

def engineer(prepared, card, state):
    hist = state['history']
    season = int(card.get('seasonality', 7) or 7)
    state['season'] = season
    state['scale'] = np.array([seasonal_scale(hist[d], season) for d in range(hist.shape[0])])
    state['anchor'] = hist[:, -min(ANCHOR_DAYS, hist.shape[1]):].mean(axis=1)
    return (prepared, state)

def select_context(variables, card, state):
    C = min(int(variables['cutoff_index']), int(card['limits']['max_context']))
    H = int(variables['horizon'])
    po = np.asarray(variables['past_features'])
    pf = np.asarray(variables['known_features'])
    case = {'target_indices': list(range(state['N'])), 'targets': np.asarray(variables['target_history'], float)[:, -C:], 'past_only': po[:, -C:] if po.size else None, 'known_future': pf[:, -(C + H):] if pf.size else None, 'past_names': variables['past_names'], 'known_names': variables['known_names'], 'provenance': 'Reference inputs: all targets/covariates, max_context=15360'}
    state['C'] = C
    return ([case], state)

def postprocess(outputs, cases, state):
    N, H = (state['N'], state['H'])
    point = np.empty((N, H))
    quantiles = np.empty((N, H, 9))
    for output, case in zip(outputs, cases):
        idx = case['target_indices']
        point[idx] = np.asarray(output['point'], float)
        quantiles[idx] = np.asarray(output['quantiles'], float)
    shift = drift_shrink_shift(point, state['anchor'], DRIFT_GAMMA, RAMP_START, RAMP_END, LEVEL_WINDOW)
    point = point + shift
    quantiles = quantiles + shift[..., None]
    quantiles = tail_widen(quantiles, TAIL_WIDEN, TAIL_START)
    point, quantiles = envelope_guard(point, quantiles, state['history'], state['scale'], ENV_BASE, ENV_GROWTH, ENV_STRENGTH)
    quantiles = np.sort(quantiles, axis=-1)
    return {'point': point, 'quantiles': quantiles}

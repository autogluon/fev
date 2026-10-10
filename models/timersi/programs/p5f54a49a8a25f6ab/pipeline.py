import numpy as np
from trend import weekly_log_growth, trend_factor
from calibrate import contract_multiplicative
DAMPING = 0.45
GROWTH_CAP = 0.4
GAMMA = 0.8
TREND_WINDOW = 21

def preprocess(view, card):
    hist = np.asarray(view['target_history'], float)
    obs = np.asarray(view.get('target_observed', np.ones_like(hist)), bool)
    clean = np.where(obs & np.isfinite(hist), hist, np.nan)
    state = {'N': len(view['target_ids']), 'H': view['horizon'], 'season': int(card.get('seasonality') or 7), 'clean': clean}
    return (view, state)

def engineer(prepared, card, state):
    season = state['season']
    growth = []
    for row in state['clean']:
        series = row[np.isfinite(row)]
        growth.append(weekly_log_growth(series, TREND_WINDOW, season))
    state['growth'] = np.asarray(growth, float)
    return (prepared, state)

def select_context(variables, card, state):
    C = min(variables['cutoff_index'], card['limits']['max_context'])
    H = variables['horizon']
    po = variables['past_features']
    pf = variables['known_features']
    return ([{'target_indices': list(range(state['N'])), 'targets': variables['target_history'][:, -C:], 'past_only': po[:, -C:] if len(po) else None, 'known_future': pf[:, -(C + H):] if len(pf) else None, 'past_names': variables['past_names'], 'known_names': variables['known_names'], 'provenance': 'Native raw inputs, all targets/covariates, max_context=15360'}], state)

def postprocess(outputs, cases, state):
    N, H, season = (state['N'], state['H'], state['season'])
    point = np.empty((N, H))
    quantiles = np.empty((N, H, 9))
    for output, case in zip(outputs, cases):
        idx = list(case['target_indices'])
        p = np.asarray(output['point'], float)
        q = np.sort(np.asarray(output['quantiles'], float), axis=-1)
        for row, target in enumerate(idx):
            fac = trend_factor(state['growth'][target], H, DAMPING, GROWTH_CAP, season)
            qt = contract_multiplicative(q[row] * fac[:, None], GAMMA)
            quantiles[target] = qt
            pt = p[row] * fac
            point[target] = np.clip(np.maximum(pt, 0.0), qt[:, 0], qt[:, 8])
    return {'point': point, 'quantiles': quantiles}

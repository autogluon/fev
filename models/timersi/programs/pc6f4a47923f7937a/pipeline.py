import numpy as np
import calendar_features as cal
import drift
import uncertainty as unc

def preprocess(view, card):
    hist = np.asarray(view['target_history'], float)
    obs = np.asarray(view['target_observed'], bool) if len(view.get('target_observed', [])) else None
    stats = [unc.own_log_volatility(hist[d], obs[d] if obs is not None else None) for d in range(hist.shape[0])]
    return (view, {'N': hist.shape[0], 'H': view['horizon'], 'stats': stats, 'hist': hist, 'obs': obs})

def engineer(prepared, card, state):
    C = int(prepared['cutoff_index'])
    H = int(prepared['horizon'])
    rows, names = cal.build(prepared['timestamps'], C, H)
    kf = prepared['known_features']
    variables = dict(prepared)
    if len(kf):
        kf = np.asarray(kf, float)
        variables['known_features'] = np.vstack([kf, rows])
        variables['known_names'] = list(prepared['known_names']) + names
    else:
        variables['known_features'] = rows
        variables['known_names'] = names
    return (variables, state)

def select_context(variables, card, state):
    C = min(variables['cutoff_index'], card['limits']['max_context'])
    H = variables['horizon']
    po = variables['past_features']
    pf = np.asarray(variables['known_features'], float)
    return ([{'target_indices': list(range(state['N'])), 'targets': np.asarray(variables['target_history'], float)[:, -C:], 'past_only': po[:, -C:] if len(po) else None, 'known_future': pf[:, -(C + H):] if len(pf) else None, 'past_names': variables['past_names'], 'known_names': variables['known_names'], 'provenance': 'Reference targets + constructed calendar trend known-covariate'}], state)

def postprocess(outputs, cases, state):
    N, H = (state['N'], state['H'])
    point = np.empty((N, H))
    quantiles = np.empty((N, H, 9))
    for output, case in zip(outputs, cases):
        pts = np.asarray(output['point'], float)
        qs = np.asarray(output['quantiles'], float)
        for j, idx in enumerate(case['target_indices']):
            s1, nd = state['stats'][idx]
            hist = state['hist'][idx]
            obs = state['obs'][idx] if state['obs'] is not None else None
            last = hist[np.isfinite(hist)][-1] if np.isfinite(hist).any() else np.nan
            med = drift.adjust_median(qs[j][:, 4], last, hist, obs)
            quantiles[idx] = unc.recalibrate(qs[j], med, s1, nd)
            point[idx] = med
    return {'point': point, 'quantiles': quantiles}

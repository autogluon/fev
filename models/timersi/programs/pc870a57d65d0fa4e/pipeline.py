import numpy as np
from transforms import choose_transform
from statmodel import fit_drift, drift_quantiles, QUANTILE_LEVELS
from combine import sanitize, clamp_to_plausible, blend_quantiles, moment_mix
CONFIG = {'anchor_weight': 0.5, 'spread_weight': None, 'disagreement': 1.0, 'combiner': 'moment_mix', 'damp': 1.0, 'half_life': 8.0, 'sigma_mult': 1.0, 'clamp_sigma': 8.0, 'feed_transformed': True, 'feed_mode': 'log_detrend'}

def _fill(z):
    z = np.asarray(z, dtype=float).copy()
    bad = ~np.isfinite(z)
    if bad.all():
        return np.zeros_like(z)
    if bad.any():
        idx = np.arange(z.size)
        z[bad] = np.interp(idx[bad], idx[~bad], z[~bad])
    return z

def preprocess(view, card):
    hist = np.asarray(view['target_history'], dtype=float)
    obs = np.asarray(view.get('target_observed', np.ones_like(hist, dtype=bool)))
    D, L = hist.shape
    transforms, z_hist = ([], [])
    for d in range(D):
        row = hist[d].copy()
        mask = obs[d].astype(bool) if obs.shape == hist.shape else np.ones(L, bool)
        row[~mask] = np.nan
        tr = choose_transform(row[np.isfinite(row)])
        transforms.append(tr)
        z_hist.append(_fill(tr.forward(np.where(np.isfinite(row), row, np.nan))))
    state = {'N': D, 'H': int(view['horizon']), 'L': int(view['cutoff_index']), 'transforms': transforms, 'z_hist': np.asarray(z_hist, dtype=float), 'raw_hist': hist, 'cfg': dict(CONFIG), 'item': view.get('item_id')}
    return (view, state)

def engineer(prepared, card, state):
    fits = [fit_drift(state['z_hist'][d], half_life=state['cfg']['half_life']) for d in range(state['N'])]
    state['fits'] = fits
    return (prepared, state)

def select_context(variables, card, state):
    L = int(variables['cutoff_index'])
    C = int(min(L, card['limits']['max_context']))
    H = int(variables['horizon'])
    if state['cfg']['feed_transformed']:
        targets = np.array(state['z_hist'][:, -C:], dtype=float)
    else:
        targets = np.asarray(variables['target_history'], float)[:, -C:]
    state['trend_offset'] = np.zeros((state['N'], H))
    if state['cfg']['feed_transformed'] and state['cfg']['feed_mode'] == 'log_detrend':
        t = np.arange(C, dtype=float)
        for d in range(state['N']):
            g = state['fits'][d]['drift']
            targets[d] = targets[d] - g * t
            state['trend_offset'][d] = g * (C + np.arange(H, dtype=float))
    targets = np.asarray(targets, dtype=float)
    if not np.isfinite(targets).all():
        targets = np.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)
    po = variables['past_features']
    pf = variables['known_features']
    case = {'target_indices': list(range(state['N'])), 'targets': targets, 'past_only': po[:, -C:] if len(po) else None, 'known_future': pf[:, -(C + H):] if len(pf) else None, 'past_names': variables['past_names'], 'known_names': variables['known_names'], 'provenance': 'log-space targets (reversible), full context %d, anchor blend' % C}
    return ([case], state)

def postprocess(outputs, cases, state):
    D, H = (state['N'], state['H'])
    cfg = state['cfg']
    point = np.empty((D, H))
    quantiles = np.empty((D, H, 9))
    for output, case in zip(outputs, cases):
        nat_p = np.asarray(output['point'], dtype=float)
        nat_q = np.asarray(output['quantiles'], dtype=float)
        for j, d in enumerate(case['target_indices']):
            fit = state['fits'][d]
            anchor = drift_quantiles(fit, H, damp=cfg['damp'], sigma_mult=cfg['sigma_mult'])
            zq = nat_q[j] if cfg['feed_transformed'] else state['transforms'][d].forward(nat_q[j])
            zq = np.asarray(zq, dtype=float) + state['trend_offset'][d][:, None]
            zq = sanitize(zq, anchor)
            zq = np.sort(zq, axis=-1)
            zq = clamp_to_plausible(zq, fit, H, n_sigma=cfg['clamp_sigma'])
            if cfg['combiner'] == 'moment_mix':
                blended = moment_mix(zq, anchor, cfg['anchor_weight'], cfg['spread_weight'], cfg['disagreement'])
            else:
                blended = blend_quantiles(zq, anchor, cfg['anchor_weight'])
            y_q = state['transforms'][d].inverse(blended)
            y_q = np.sort(np.asarray(y_q, dtype=float), axis=-1)
            quantiles[d] = y_q
            point[d] = y_q[:, 4]
            if not np.isfinite(quantiles[d]).all() or not np.isfinite(point[d]).all():
                fb = state['transforms'][d].inverse(anchor)
                quantiles[d] = np.sort(fb, axis=-1)
                point[d] = quantiles[d][:, 4]
    return {'point': point, 'quantiles': quantiles}

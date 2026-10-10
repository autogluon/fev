import numpy as np
try:
    from . import climatology as clim
    from . import leveladjust as lvl
    from . import grouping as grp
    from . import calibration as cal
except ImportError:
    import climatology as clim
    import leveladjust as lvl
    import grouping as grp
    import calibration as cal

def _as_array(x):
    a = np.asarray(x)
    return a

def preprocess(view, card):
    state = {'N': len(view['target_ids']), 'H': int(view['horizon']), 'L': int(view['cutoff_index']), 'item': view.get('item_id'), 'diag': []}
    return (view, state)

def engineer(prepared, card, state):
    view = dict(prepared)
    hist = np.array(view['target_history'], dtype=float)
    L, H = (state['L'], state['H'])
    ts = _as_array(view['timestamps'])
    s_all = clim.climate_basis(ts)
    s_past, s_fut = (s_all[:L], s_all[L:L + H])
    names = list(view['target_names'])
    comp_future = np.zeros((state['N'], H))
    adjusted = hist.copy()
    for i, name in enumerate(names):
        if not clim.is_temperature_target(name):
            continue
        y = hist[i]
        obs = np.asarray(view['target_observed'], dtype=bool)[i] if 'target_observed' in view else np.ones(L, bool)
        mask = obs & np.isfinite(y)
        if mask.sum() < 6:
            continue
        slope, kappa, diag = clim.fit_seasonal(y[mask], s_past[mask])
        diag['target'] = name
        state['diag'].append(diag)
        if slope == 0.0:
            continue
        s_ref = 0.5 * float(s_past[mask].mean()) + 0.5 * float(s_past[mask][-1])
        adjusted[i] = y - slope * (s_past - s_ref)
        comp_future[i] = slope * (s_fut - s_ref)
    view['target_history'] = adjusted
    state['comp_future'] = comp_future
    state['level_shift'] = lvl.local_level_shift(adjusted)
    state['trend'] = lvl.damped_trend(adjusted, H)
    state['groups'] = grp.target_groups(names)
    return (view, state)

def select_context(variables, card, state):
    C = min(int(variables['cutoff_index']), int(card['limits']['max_context']))
    H = int(variables['horizon'])
    po = _as_array(variables['past_features'])
    pf = _as_array(variables['known_features'])
    hist = np.asarray(variables['target_history'], dtype=float)
    groups = state.get('groups') or [list(range(state['N']))]
    cases = []
    for g in groups:
        cases.append({'target_indices': list(g), 'targets': hist[g][:, -C:], 'past_only': po[:, -C:] if len(po) else None, 'known_future': pf[:, -(C + H):] if len(pf) else None, 'past_names': variables['past_names'], 'known_names': variables['known_names'], 'provenance': 'Own past, OT climatologically de-seasonalised; targets grouped by voltage family (main H/M+OT vs low)'})
    return (cases, state)

def postprocess(outputs, cases, state):
    N, H = (state['N'], state['H'])
    point = np.empty((N, H))
    quantiles = np.empty((N, H, 9))
    for output, case in zip(outputs, cases):
        point[case['target_indices']] = output['point']
        quantiles[case['target_indices']] = output['quantiles']
    delta = np.zeros((N, H))
    comp = state.get('comp_future')
    if comp is not None:
        delta = delta + comp
    shift = state.get('level_shift')
    if shift is not None:
        delta = delta + np.asarray(shift, dtype=float)[:, None]
    trend = state.get('trend')
    if trend is not None:
        delta = delta + np.asarray(trend, dtype=float)
    point = point + delta
    quantiles = quantiles + delta[:, :, None]
    quantiles = cal.rescale_spread(quantiles)
    return {'point': point, 'quantiles': quantiles, 'components': {'climatology': np.asarray(state.get('comp_future')), 'level_shift': np.asarray(state.get('level_shift')), 'damped_trend': np.asarray(state.get('trend'))}}

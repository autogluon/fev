import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F

def preprocess(view, card):
    state = {'N': len(view['target_ids']), 'H': view['horizon'], 'L': view['cutoff_index'], 'item': view.get('item_id')}
    state['dates'] = F.to_days([str(t)[:10] for t in view['timestamps']])
    return (view, state)

def engineer(prepared, card, state):
    L, H = (state['L'], state['H'])
    dates = state['dates']
    y = np.asarray(prepared['target_history'], float)[0]
    obs = np.asarray(prepared['target_observed'], float)[0] if len(prepared.get('target_observed', [])) else None
    if obs is not None:
        y = np.where(obs > 0, y, np.nan)
    good = np.isfinite(y) & (y > 0)
    if good.sum() < 400:
        state['cal'] = None
        return (prepared, state)
    yf = np.where(good, y, np.nan)
    idx = np.arange(len(yf))
    yf = np.interp(idx, idx[good], yf[good])
    cal = F.CalendarModel(yf, dates[:L])
    if not cal.ok:
        state['cal'] = None
        return (prepared, state)
    rows = np.stack([cal.dow_feature(dates), cal.holiday_feature(dates), cal.annual_feature(dates)])
    state['cal'] = cal
    state['extra_known'] = rows
    state['extra_names'] = ['cal_dow_log', 'cal_holiday_log', 'cal_annual_log']
    return (prepared, state)

def select_context(variables, card, state):
    C = min(variables['cutoff_index'], card['limits']['max_context'])
    H = variables['horizon']
    po = variables['past_features']
    pf = np.asarray(variables['known_features'], float) if len(variables['known_features']) else None
    names = list(variables['known_names'])
    extra = state.get('extra_known')
    if extra is not None:
        block = np.asarray(extra, float)[:, -(C + H):]
        if not np.all(np.isfinite(block)):
            block = np.nan_to_num(block, nan=0.0, posinf=0.0, neginf=0.0)
        kf = block if pf is None else np.concatenate([pf[:, -(C + H):], block], 0)
        names = names + state['extra_names']
        prov = 'full context + 3 causal calendar regressors (dow/holiday/annual, log-additive)'
    else:
        kf = None if pf is None else pf[:, -(C + H):]
        prov = 'full context, calendar model unavailable'
    return ([{'target_indices': list(range(state['N'])), 'targets': np.asarray(variables['target_history'], float)[:, -C:], 'past_only': po[:, -C:] if len(po) else None, 'known_future': kf, 'past_names': variables['past_names'], 'known_names': names, 'provenance': prov}], state)

def postprocess(outputs, cases, state):
    point = np.empty((state['N'], state['H']))
    quantiles = np.empty((state['N'], state['H'], 9))
    for output, case in zip(outputs, cases):
        point[case['target_indices']] = output['point']
        quantiles[case['target_indices']] = output['quantiles']
    quantiles = np.sort(quantiles, axis=-1)
    return {'point': point, 'quantiles': quantiles}

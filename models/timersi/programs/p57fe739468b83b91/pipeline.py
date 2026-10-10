import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from regime import detect_regime_start
from calendarx import day_type, temp_channels
from causal import causal_estimate
TEMP_NAMES = ('airtemperature', 'temperature', 'temp')

def preprocess(view, card):
    state = {'N': len(view['target_ids']), 'H': view['horizon'], 'item': view.get('item_id'), 'notes': {}}
    return (view, state)

def engineer(prepared, card, state):
    y = np.asarray(prepared['target_history'], dtype=float)
    obs = np.asarray(prepared['target_observed'], dtype=bool)
    L = int(prepared['cutoff_index'])
    state['regime_starts'] = [detect_regime_start(y[d, :L], obs[d, :L]) for d in range(state['N'])]
    state['notes']['regime'] = {prepared['target_ids'][d]: int(s) for d, s in enumerate(state['regime_starts']) if s > 0}
    ts = np.asarray(prepared['timestamps'], dtype='datetime64[h]')
    known = np.asarray(prepared['known_features'], dtype=float) if len(prepared['known_features']) else np.zeros((0, ts.size))
    names = list(prepared['known_names'])
    extra, extra_names = ([], [])
    try:
        extra.append(day_type(ts))
        extra_names.append('cal_day_type')
    except Exception:
        extra, extra_names = ([], [])
    ti = next((i for i, nm in enumerate(names) if str(nm).lower() in TEMP_NAMES), None)
    if ti is not None and known.shape[1] == ts.size:
        T = known[ti]
        if np.isfinite(T).all():
            ma24, hdd, cdd, _ = temp_channels(T)
            extra += [hdd, cdd, (ma24 - 65.0) / 25.0]
            extra_names += ['thermal_hdd', 'thermal_cdd', 'thermal_ma24']
    if extra:
        add = np.asarray(extra, dtype=float)
        if np.isfinite(add).all() and add.shape[1] == ts.size:
            known = np.vstack([known, add]) if known.size else add
            names = names + extra_names
    chan, chan_sd = ({}, {})
    if ti is not None and known.shape[1] == ts.size and np.isfinite(known[ti]).all():
        for d in range(state['N']):
            try:
                res = causal_estimate(ts, known[ti], y[d, :L], obs[d, :L], L, start=state['regime_starts'][d])
            except Exception:
                res = None
            if res is not None and np.isfinite(res[0]).all():
                chan[d] = np.asarray(res[0], dtype=float)
                chan_sd[d] = res[1]
    state['causal_chan'] = chan
    state['known_block'] = known
    state['known_names'] = names
    state['notes']['known_names'] = names
    state['notes']['causal_fit_sd'] = {prepared['target_ids'][d]: round(v, 4) for d, v in chan_sd.items()}
    return (prepared, state)

def select_context(variables, card, state):
    L = int(variables['cutoff_index'])
    H = int(variables['horizon'])
    cap = int(card['limits']['max_context'])
    po = variables['past_features']
    pf = state['known_block']
    chan = state['causal_chan']
    groups = {}
    for d in range(state['N']):
        key = (int(min(L - state['regime_starts'][d], cap)), d in chan)
        groups.setdefault(key, []).append(d)
    cases = []
    for (C, has), idx in sorted(groups.items(), key=lambda kv: -kv[0][0]):
        names = list(state['known_names'])
        blocks = [pf] if pf.shape[0] else []
        if has:
            blocks.append(np.asarray([chan[d] for d in idx], dtype=float))
            names += ['causal_load_est_%d' % d for d in idx]
        kf = np.vstack(blocks)[:, L - C:L + H] if blocks else None
        cases.append({'target_indices': list(idx), 'targets': np.asarray(variables['target_history'], dtype=float)[idx, L - C:L], 'past_only': np.asarray(po, dtype=float)[:, L - C:L] if len(po) else None, 'known_future': kf, 'past_names': variables['past_names'], 'known_names': names, 'provenance': 'regime-trimmed C=%d, known=%s' % (C, ','.join(names))})
    return (cases, state)

def postprocess(outputs, cases, state):
    point = np.empty((state['N'], state['H']))
    quantiles = np.empty((state['N'], state['H'], 9))
    for output, case in zip(outputs, cases):
        point[case['target_indices']] = output['point']
        quantiles[case['target_indices']] = output['quantiles']
    return {'point': point, 'quantiles': quantiles}

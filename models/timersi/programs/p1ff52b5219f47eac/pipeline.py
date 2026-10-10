import numpy as np
from structure import annual_structure
from calendarfeat import fourier
import backtest as bt
from external_channel import level_channel
PERIOD = 52

def _finite(a, fill=0.0):
    a = np.asarray(a, dtype=float)
    return np.where(np.isfinite(a), a, fill)

def preprocess(view, card):
    hist = _finite(view['target_history'])
    obs = np.asarray(view['target_observed'])
    for d in range(hist.shape[0]):
        row, mask = (hist[d], obs[d].astype(bool))
        if not mask.all() and mask.any():
            last = row[mask][0]
            for t in range(len(row)):
                if mask[t]:
                    last = row[t]
                else:
                    row[t] = last
    prepared = dict(view)
    prepared['target_history'] = hist
    state = {'N': hist.shape[0], 'H': int(view['horizon']), 'L': int(view['cutoff_index']), 'item_id': view.get('item_id')}
    return (prepared, state)

def _channels(hist, timestamps, official, official_names, Lc, H, N):
    total = Lc + H
    extra, extra_names, diag = ([], [], [])
    for d in range(N):
        try:
            s = annual_structure(hist[d][:Lc], total, period=PERIOD)
        except Exception as exc:
            diag.append({'error': repr(exc)})
            continue
        extra.append(_finite(s['struct']))
        extra_names.append('struct_est_%d' % d)
        extra.append(_finite(s['season'], 1.0))
        extra_names.append('season_shape_%d' % d)
        diag.append(s['info'])
    if len(official):
        for r, nm in zip(official, official_names):
            ch = level_channel(np.asarray(r, float)[:total], hist[0][:Lc], Lc)
            if ch is not None:
                extra.append(_finite(ch))
                extra_names.append('%s_level_est' % nm)
    cal, cal_names = fourier(timestamps[:total], harmonics=(1,))
    extra.extend(list(cal))
    extra_names.extend(cal_names)
    rows = np.array(extra) if len(extra) else np.zeros((0, total))
    allknown = np.vstack([official, rows]) if len(official) else rows
    return (_finite(allknown), list(official_names) + extra_names, diag)

def engineer(prepared, card, state):
    L, H = (state['L'], state['H'])
    total = L + H
    hist = prepared['target_history']
    known = _finite(prepared['known_features']) if len(prepared['known_features']) else np.zeros((0, total))
    feats, names, diag = _channels(hist, prepared['timestamps'], known, list(prepared['known_names']), L, H, state['N'])
    variables = dict(prepared)
    variables['known_features'] = feats
    variables['known_names'] = names
    state['diag'] = diag
    state['official_known'] = known
    state['official_names'] = list(prepared['known_names'])
    return (variables, state)

def select_context(variables, card, state):
    C = min(int(variables['cutoff_index']), int(card['limits']['max_context']))
    H = int(variables['horizon'])
    po = variables['past_features']
    pf = variables['known_features']
    cases = [{'target_indices': list(range(state['N'])), 'targets': variables['target_history'][:, -C:], 'past_only': np.asarray(po)[:, -C:] if len(po) else None, 'known_future': np.asarray(pf)[:, -(C + H):] if len(pf) else None, 'past_names': list(variables['past_names']), 'known_names': list(variables['known_names']), 'provenance': 'official external + context-aware own-past annual structure'}]
    L = int(variables['cutoff_index'])
    hist = variables['target_history']
    maxc = int(card['limits']['max_context'])
    state['backtests'] = []
    for o in bt.plan_origins(L, H):
        Lc = L - o
        Cb = min(Lc, maxc)
        if Cb < 1 or Lc + H > L:
            continue
        try:
            feats, names, _ = _channels(hist, variables['timestamps'], state['official_known'][:, :Lc + H] if len(state['official_known']) else state['official_known'], state['official_names'], Lc, H, state['N'])
        except Exception:
            continue
        cases.append({'target_indices': list(range(state['N'])), 'targets': hist[:, Lc - Cb:Lc], 'past_only': np.asarray(po)[:, Lc - Cb:Lc] if len(po) else None, 'known_future': feats[:, -(Cb + H):] if len(feats) else None, 'past_names': list(variables['past_names']), 'known_names': names, 'auxiliary': True, 'origin_offset': int(o), 'provenance': 'native historical backtest, origin_offset=%d, C=%d' % (o, Cb)})
        state['backtests'].append({'offset': int(o), 'context': int(Cb), 'truth': hist[:, Lc:Lc + H].copy()})
    return (cases, state)

def postprocess(outputs, cases, state):
    N, H = (state['N'], state['H'])
    point = np.zeros((N, H))
    quantiles = np.zeros((N, H, 9))
    records = []
    bidx = 0
    for output, case in zip(outputs, cases):
        if case.get('auxiliary'):
            if bidx < len(state.get('backtests', [])):
                meta = state['backtests'][bidx]
                qb = _finite(output['quantiles'])
                for j, d in enumerate(case['target_indices']):
                    records.append({'q': np.sort(qb[j], axis=-1), 'truth': np.asarray(meta['truth'][d], float), 'context': meta['context'], 'target': int(d)})
            bidx += 1
            continue
        point[case['target_indices']] = _finite(output['point'])
        quantiles[case['target_indices']] = _finite(output['quantiles'])
    quantiles = np.sort(quantiles, axis=2)
    cal = {'shift': 0.0, 'wlo': 1.0, 'whi': 1.0, 'diag': {'n_records': 0}}
    try:
        cal = bt.measure(records)
        for d in range(N):
            own = [r for r in records if r['target'] == d]
            c = bt.measure(own) if own else cal
            point[d], quantiles[d] = bt.apply_calibration(point[d], quantiles[d], c)
    except Exception as exc:
        cal['error'] = repr(exc)
    return {'point': point, 'quantiles': quantiles, 'components': {'diagnostics': state.get('diag', []), 'calibration': {k: v for k, v in cal.items() if k != 'diag'}, 'calibration_diag': cal.get('diag', {})}}

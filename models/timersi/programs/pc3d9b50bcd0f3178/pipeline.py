import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import expert as EX
except ImportError:
    from . import expert as EX
CONFIG = {'calendar_cov': False, 'expert_cov': False, 'expert_blend': 0.5, 'spread': 1.0, 'n_origins': 520}

def _calendar_covariates(ts):
    cal = EX.calendar_frame(ts)
    slot = cal['slot'].astype(np.float64)
    dow = cal['dow'].astype(np.float64)
    doy = cal['doy'].astype(np.float64)
    rows = [np.sin(2 * np.pi * slot / 96.0), np.cos(2 * np.pi * slot / 96.0), np.sin(2 * np.pi * dow / 7.0), np.cos(2 * np.pi * dow / 7.0), np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25), EX.holiday_flag(cal).astype(np.float64) + EX.xmas_window(cal).astype(np.float64)]
    names = ['cal_tod_sin', 'cal_tod_cos', 'cal_dow_sin', 'cal_dow_cos', 'cal_doy_sin', 'cal_doy_cos', 'cal_offday']
    return (np.asarray(rows), names)

def preprocess(view, card):
    state = {'N': len(view['target_ids']), 'H': view['horizon'], 'cfg': dict(CONFIG), 'notes': []}
    return (view, state)

def engineer(prepared, card, state):
    v = dict(prepared)
    cfg = state['cfg']
    ts = np.asarray(v['timestamps'], dtype='datetime64[ns]')
    known = np.asarray(v['known_features'], dtype=np.float64) if len(v['known_features']) else np.zeros((0, len(ts)))
    names = list(v['known_names'])
    extra, extra_names = ([], [])
    if cfg['calendar_cov']:
        try:
            rows, nms = _calendar_covariates(ts)
            extra.append(rows)
            extra_names += nms
            state['notes'].append('calendar covariates added')
        except Exception as exc:
            state['notes'].append('calendar covariates failed: %r' % (exc,))
    need_expert = cfg['expert_cov'] or cfg['expert_blend'] > 0
    if need_expert:
        try:
            y = np.asarray(v['target_history'], dtype=np.float64)[0]
            pred, _, _ = EX.fit_predict(y, ts, known if len(known) else None, int(v['cutoff_index']), int(v['horizon']), n_origins=cfg['n_origins'])
            state['expert'] = pred
            state['notes'].append('expert fitted')
            if cfg['expert_cov']:
                L, H = (int(v['cutoff_index']), int(v['horizon']))
                row = np.empty(L + H)
                row[:L] = y
                row[L:] = pred
                extra.append(row[None, :])
                extra_names.append('causal_dayahead_estimate')
        except Exception as exc:
            state['notes'].append('expert failed: %r' % (exc,))
            state['expert'] = None
    if extra:
        allrows = np.concatenate([known] + extra, axis=0) if len(known) else np.concatenate(extra, axis=0)
        allrows = np.nan_to_num(allrows, nan=0.0, posinf=0.0, neginf=0.0)
        v['known_features'] = allrows
        v['known_names'] = names + extra_names
    return (v, state)

def select_context(variables, card, state):
    C = min(variables['cutoff_index'], card['limits']['max_context'])
    H = variables['horizon']
    po = np.asarray(variables['past_features'])
    pf = np.asarray(variables['known_features'])
    return ([{'target_indices': list(range(state['N'])), 'targets': np.asarray(variables['target_history'])[:, -C:], 'past_only': po[:, -C:] if len(po) else None, 'known_future': pf[:, -(C + H):] if len(pf) else None, 'past_names': variables['past_names'], 'known_names': variables['known_names'], 'provenance': 'all targets, official + constructed calendar covariates, C=%d' % C}], state)

def postprocess(outputs, cases, state):
    N, H = (state['N'], state['H'])
    point = np.empty((N, H))
    quantiles = np.empty((N, H, 9))
    for output, case in zip(outputs, cases):
        point[case['target_indices']] = output['point']
        quantiles[case['target_indices']] = output['quantiles']
    cfg = state['cfg']
    w = cfg['expert_blend']
    exp = state.get('expert')
    if w > 0 and exp is not None and (N == 1) and np.all(np.isfinite(exp)):
        med = quantiles[0, :, 4]
        shift = w * (np.asarray(exp, dtype=np.float64) - med)
        new_med = med + shift
        spread = cfg.get('spread', 1.0)
        quantiles[0] = new_med[:, None] + spread * (quantiles[0] - med[:, None])
        point[0] = point[0] + shift
        state['notes'].append('expert blend w=%.2f applied' % w)
    quantiles = np.sort(quantiles, axis=2)
    return {'point': point, 'quantiles': quantiles, 'components': {'notes': state['notes']}}

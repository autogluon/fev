import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import expert as EX
except ImportError:
    from . import expert as EX
CONFIG = {'cal_cov': True, 'hdd_cdd': True, 'aux_recenter': {'enabled': True, 'offsets': [96, 192], 'shrink': 0.5, 'cap': 0.04}, 'expert_blend': 0.0}

def _calendar_covariates(ts):
    cal = EX.calendar_frame(ts)
    slot = cal['slot'].astype(np.float64)
    dow = cal['dow'].astype(np.float64)
    doy = cal['doy'].astype(np.float64)
    offday = EX.holiday_flag(cal).astype(np.float64)
    offday = np.maximum(offday, EX.xmas_window(cal).astype(np.float64))
    rows = [np.sin(2 * np.pi * slot / 96.0), np.cos(2 * np.pi * slot / 96.0), np.sin(2 * np.pi * dow / 7.0), np.cos(2 * np.pi * dow / 7.0), np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25), (dow >= 5).astype(np.float64), offday]
    names = ['cal_tod_sin', 'cal_tod_cos', 'cal_dow_sin', 'cal_dow_cos', 'cal_doy_sin', 'cal_doy_cos', 'cal_weekend', 'cal_offday']
    return (np.asarray(rows), names)

def preprocess(view, card):
    state = {'N': len(view['target_ids']), 'H': view['horizon'], 'cfg': dict(CONFIG), 'notes': [], 'components': {}}
    return (view, state)

def engineer(prepared, card, state):
    v = dict(prepared)
    cfg = state['cfg']
    ts = np.asarray(v['timestamps'], dtype='datetime64[ns]')
    known = np.asarray(v['known_features'], dtype=np.float64) if len(v['known_features']) else np.zeros((0, len(ts)))
    names = list(v['known_names'])
    extra, extra_names = ([], [])
    if cfg['cal_cov']:
        try:
            rows, nms = _calendar_covariates(ts)
            extra.append(rows)
            extra_names += nms
            state['notes'].append('calendar covariates added')
        except Exception as exc:
            state['notes'].append('calendar covariates failed: %r' % (exc,))
    if cfg['hdd_cdd'] and 'temperature' in names:
        t = known[names.index('temperature')]
        extra.append(np.stack([np.maximum(16.0 - t, 0.0), np.maximum(t - 20.0, 0.0)]))
        extra_names += ['temp_hdd16', 'temp_cdd20']
        state['notes'].append('hdd/cdd added')
    if extra:
        allrows = np.concatenate([known] + extra, axis=0) if len(known) else np.concatenate(extra, axis=0)
        allrows = np.nan_to_num(allrows, nan=0.0, posinf=0.0, neginf=0.0)
        v['known_features'] = allrows
        v['known_names'] = names + extra_names
    state['components']['constructed_known_covariates'] = extra_names
    return (v, state)

def select_context(variables, card, state):
    L = int(variables['cutoff_index'])
    H = int(variables['horizon'])
    C = min(L, card['limits']['max_context'])
    po = np.asarray(variables['past_features'])
    pf = np.asarray(variables['known_features'])
    tg = np.asarray(variables['target_history'])
    cases = [{'target_indices': list(range(state['N'])), 'targets': tg[:, L - C:L], 'past_only': po[:, L - C:L] if len(po) else None, 'known_future': pf[:, L - C:L + H] if len(pf) else None, 'past_names': variables['past_names'], 'known_names': variables['known_names'], 'provenance': 'primary, official + constructed semantic covariates, C=%d' % C}]
    state['aux_truths'] = []
    rc = state['cfg']['aux_recenter']
    if rc['enabled']:
        for off in rc['offsets']:
            end = L - off
            if end < 2:
                continue
            Ca = min(end, card['limits']['max_context'])
            cases.append({'target_indices': list(range(state['N'])), 'targets': tg[:, end - Ca:end], 'past_only': po[:, end - Ca:end] if len(po) else None, 'known_future': pf[:, end - Ca:end + H] if len(pf) else None, 'past_names': variables['past_names'], 'known_names': variables['known_names'], 'auxiliary': True, 'origin_offset': off, 'provenance': 'auxiliary native backtest, offset=%d, C=%d' % (off, Ca)})
            state['aux_truths'].append(tg[:, end:end + H])
    return (cases, state)

def postprocess(outputs, cases, state):
    N, H = (state['N'], state['H'])
    point = np.empty((N, H))
    quantiles = np.empty((N, H, 9))
    aux_meds = []
    for output, case in zip(outputs, cases):
        if case.get('auxiliary'):
            aux_meds.append(np.asarray(output['quantiles'])[:, :, 4])
            continue
        point[case['target_indices']] = output['point']
        quantiles[case['target_indices']] = output['quantiles']
    cfg = state['cfg']
    rc = cfg['aux_recenter']
    if rc['enabled'] and aux_meds and state['aux_truths']:
        biases = np.stack([np.mean(tr - md, axis=1) for tr, md in zip(state['aux_truths'], aux_meds)])
        level = np.stack([np.maximum(np.mean(np.abs(tr), axis=1), 1e-09) for tr in state['aux_truths']]).mean(axis=0)
        agree = (np.sign(biases) == np.sign(biases[0:1])).all(axis=0) & (np.abs(biases) > 0).all(axis=0)
        shift = rc['shrink'] * biases.mean(axis=0) * agree
        shift = np.clip(shift, -rc['cap'] * level, rc['cap'] * level)
        point += shift[:, None]
        quantiles += shift[:, None, None]
        state['components']['aux_rel_bias'] = (biases / level[None, :]).tolist()
        state['components']['applied_rel_shift'] = (shift / level).tolist()
    quantiles = np.sort(quantiles, axis=2)
    return {'point': point, 'quantiles': quantiles, 'components': dict(state['components'], notes=state['notes'])}

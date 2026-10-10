import importlib.util
import os
import sys
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

def _sibling(name):
    try:
        return importlib.import_module(name)
    except Exception:
        spec = importlib.util.spec_from_file_location('_fevres_' + name, os.path.join(_HERE, name + '.py'))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod
_inp = _sibling('inputs')
_cal = _sibling('calibration')
_norm = _sibling('normalize')
_season = _sibling('seasonal_risk')
_nat = _sibling('native_cal')
PRUNE_CONSTANT_COVARIATES = False
LOG_TARGET = False
N_AUX = 2
USE_LEVEL_CORRECTION = True
USE_MEASURED_SPREAD = False
USE_HEURISTIC_SHRINK = True

def preprocess(view, card):
    state = {'N': len(view['target_ids']), 'H': int(view['horizon']), 'L': int(view['cutoff_index']), 'item_id': view.get('item_id')}
    raw = np.asarray(view['target_history'], dtype=float)
    state['raw_history'] = raw
    obs = view.get('target_observed')
    state['observed'] = np.asarray(obs) if obs is not None else None
    use = _norm.plan(raw, obs) if LOG_TARGET else np.zeros(len(raw), bool)
    state['log_use'] = use
    prepared = dict(view)
    prepared['target_history'] = _norm.forward(raw, use)
    return (prepared, state)

def engineer(prepared, card, state):
    L, H = (state['L'], state['H'])
    stamps = np.asarray(prepared.get('timestamps', []))
    risks = np.zeros(state['N'])
    if stamps.size >= L + H:
        past_ts, fut_ts = (stamps[:L], stamps[L:L + H])
        for d in range(state['N']):
            try:
                risks[d], _ = _season.risk(state['raw_history'][d], past_ts, fut_ts)
            except Exception:
                risks[d] = 0.0
    state['calendar_risk'] = risks
    return (prepared, state)

def select_context(variables, card, state):
    cases = _inp.build_cases(variables, state, max_context=int(card['limits']['max_context']), prune=PRUNE_CONSTANT_COVARIATES, n_aux=N_AUX)
    return (cases, state)

def _aux_weight(meta, H):
    role = meta.get('role')
    if role == 'recent':
        return 1.0
    if role == 'seasonal':
        return 1.0
    return 0.6

def postprocess(outputs, cases, state):
    N, H = (state['N'], state['H'])
    Y = state['raw_history']
    use = state['log_use']
    meta = state.get('case_meta', [])
    point = np.empty((N, H))
    quantiles = np.empty((N, H, 9))
    aux = {d: [] for d in range(N)}
    primary = None
    for i, (output, case) in enumerate(zip(outputs, cases)):
        m = meta[i] if i < len(meta) else {'role': 'primary'}
        pt = np.asarray(output['point'], dtype=float)
        qs = np.asarray(output['quantiles'], dtype=float)
        if not case.get('auxiliary'):
            primary = (case, pt, qs)
            continue
        sl = m.get('truth_slice')
        if not sl:
            continue
        a, b = (int(sl[0]), int(sl[1]))
        for r, d in enumerate(case['target_indices']):
            truth = Y[d][a:b]
            if truth.size != pt.shape[1]:
                continue
            if state['observed'] is not None:
                if not np.all(np.asarray(state['observed'])[d][a:b].astype(bool)):
                    continue
            aux[d].append({'point': pt[r], 'quantiles': qs[r], 'truth': truth, 'weight': _aux_weight(m, H), 'role': m.get('role'), 'offset': m.get('offset')})
    case, pt, qs = primary
    notes = []
    for r, d in enumerate(case['target_indices']):
        p_d = pt[r]
        q_d = qs[r]
        res = _nat.residuals(aux[d]) if USE_LEVEL_CORRECTION else []
        b, binfo = _nat.level_correction(res, H)
        s_meas, sinfo = _nat.spread_scale(res)
        spread = s_meas if USE_MEASURED_SPREAD else 1.0
        p_d, q_d = _nat.apply(p_d, q_d, b, spread)
        s, st = (1.0, {})
        if USE_HEURISTIC_SHRINK:
            p_d, q_d, s, st = _cal.recalibrate(p_d, q_d, Y[d], H, calendar_risk=state['calendar_risk'][d])
        point[d] = _norm.inverse_point(p_d, use[d])
        quantiles[d] = _norm.inverse_quantiles(q_d, use[d])
        notes.append({'target': int(d), 'item': state.get('item_id'), 'n_aux': len(aux[d]), 'aux_roles': [a['role'] for a in aux[d]], 'aux_offsets': [a['offset'] for a in aux[d]], 'aux_mean_logres': [float(np.nanmean(np.log(np.maximum(a['truth'], 1e-09) / np.maximum(a['point'], 1e-09)))) for a in aux[d]], 'level': binfo, 'measured_spread': sinfo, 'shrink': float(s), 'state': st})
    return {'point': point, 'quantiles': quantiles, 'components': {'native_calibration': notes}}

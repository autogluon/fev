import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import structure
import activity
import cases as case_lib
Q_LEVELS = np.arange(1, 10) / 10.0
AUX_OFFSETS = (60, 300)
AUX_CONTEXT = 8192
PHASE_WEIGHT = 0.33
PHASE_WINDOWS = (150, 600)
PHASE_HALF_LIFE = 100.0
MIN_ATOM = 0.01
MIN_ACF = 0.1

def preprocess(view, card):
    Y = np.asarray(view['target_history'], dtype=np.float64)
    O = np.asarray(view['target_observed'], dtype=bool)
    summary = structure.describe_targets(Y, O)
    floor = structure.support_floor(summary, min_atom=MIN_ATOM)
    floor_val = np.where(np.isfinite(floor), floor, summary['min'])
    act = activity.activity_mask(Y, floor_val)
    period, acf, ok = activity.detect_period(act, min_acf=MIN_ACF)
    modal = int(np.bincount(period[ok]).argmax()) if ok.any() else 0
    period_used = np.where(ok, period, modal)
    state = {'N': Y.shape[0], 'H': int(view['horizon']), 'L': int(view['cutoff_index']), 'floor': floor, 'floor_val': floor_val, 'mins': summary['min'], 'atom': summary['atom'], 'pos_scale': summary['pos_scale'], 'period': period_used, 'period_acf': acf, 'phase_gate': (summary['atom'] >= MIN_ATOM) & (period_used > 1), 'modal_period': modal, 'aux_truth': {}, 'diag': {}}
    state['p_phase'] = activity.phase_probability(act, period_used, state['H'], PHASE_WINDOWS, PHASE_HALF_LIFE)
    view = dict(view)
    view['target_history'] = Y
    return (view, state)

def engineer(prepared, card, state):
    return (prepared, state)

def select_context(variables, card, state):
    cap = int(card.get('limits', {}).get('max_context', 15360))
    L, H = (state['L'], state['H'])
    out = [case_lib.primary(variables, state, min(L, cap))]
    for off in AUX_OFFSETS:
        c = case_lib.auxiliary(variables, state, off, min(AUX_CONTEXT, cap))
        if c is None:
            continue
        end = L - off
        state['aux_truth'][len(out)] = variables['target_history'][:, end:end + H]
        out.append(c)
    state['n_cases'] = len(out)
    return (out, state)

def _combine(point, quantiles, state):
    fl = state['floor_val']
    hard = np.where(np.isfinite(state['floor']), state['floor'], -np.inf)
    cl = np.sort(np.maximum(quantiles, hard[:, None, None]), axis=-1)
    p_phase = state.get('p_phase')
    if p_phase is None:
        return (np.maximum(point, hard[:, None]), cl)
    tol = 1e-09 * np.maximum(1.0, np.abs(fl))
    p_native = (cl > fl[:, None, None] + tol[:, None, None]).mean(axis=2)
    w = PHASE_WEIGHT * state['phase_gate'].astype(np.float64)[:, None]
    p_new = activity._sigmoid((1.0 - w) * activity._logit(p_native) + w * activity._logit(p_phase))
    q = activity.recompose(cl, fl, p_new, p_native, Q_LEVELS)
    q = np.sort(np.maximum(q, hard[:, None, None]), axis=-1)
    return (q[:, :, 4].copy(), q)

def postprocess(outputs, cases, state):
    N, H = (state['N'], state['H'])
    point = np.zeros((N, H))
    quantiles = np.zeros((N, H, 9))
    filled = np.zeros(N, dtype=bool)
    for idx, (output, case) in enumerate(zip(outputs, cases)):
        tgt = np.asarray(case['target_indices'], dtype=int)
        if case.get('auxiliary'):
            truth = state['aux_truth'].get(idx)
            if truth is not None:
                e = np.asarray(output['point'], dtype=np.float64) - truth[tgt]
                state['diag']['aux_%d_mae' % case.get('origin_offset', idx)] = float(np.mean(np.abs(e)))
            continue
        if case.get('alternative'):
            continue
        point[tgt] = np.asarray(output['point'], dtype=np.float64)
        quantiles[tgt] = np.asarray(output['quantiles'], dtype=np.float64)
        filled[tgt] = True
    if not filled.all():
        raise RuntimeError('primary cases did not cover all targets')
    point, quantiles = _combine(point, quantiles, state)
    return {'point': point, 'quantiles': quantiles}

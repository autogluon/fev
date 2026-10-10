import os
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calendar_es import event_features
import seasonal
EVENT_SCALE = 0.9
LOWER_SHRINK = 0.5
UPPER_SHRINK = 0.85
POINT_UPLIFT = 1.005
SPECIALIST_WEIGHT = 0.3
LEVEL_WINDOW = 21
MIN_HOLIDAYS = 3
CALENDAR_COVARIATE = True
GAP_CREDIT = 1.0
CLEAN_CONTEXT = False

def _daily(ts):
    if len(ts) < 3:
        return False
    d = np.diff(ts.values).astype('timedelta64[h]').astype(float)
    return abs(np.median(d) - 24.0) < 1e-06

def preprocess(view, card):
    ts = pd.to_datetime(view['timestamps'])
    L = int(view['cutoff_index'])
    H = int(view['horizon'])
    state = {'N': len(view['target_ids']), 'H': H, 'L': L, 'past_idx': ts[:L], 'fut_idx': ts[L:L + H], 'daily': _daily(ts)}
    return (view, state)

def engineer(prepared, card, state):
    N, H = (state['N'], state['H'])
    hist_raw = np.asarray(prepared['target_history'], float)
    state['log_adj'] = np.zeros((N, H))
    state['past_log_adj'] = np.zeros((N, state['L']))
    state['dowf'] = np.zeros((N, H))
    state['spec_log'] = np.full((N, H), np.nan)
    state['clean_history'] = hist_raw.copy()
    if not state['daily']:
        state['note'] = 'non-daily frequency: calendar module disabled'
        return (prepared, state)
    ev = event_features(pd.DatetimeIndex(state['past_idx']).union(pd.DatetimeIndex(state['fut_idx'])))
    hist = hist_raw
    obs = np.asarray(prepared['target_observed'], bool) if len(prepared['target_observed']) else None
    diag = []
    for d in range(N):
        y = hist[d]
        o = obs[d] if obs is not None else None
        try:
            dec = seasonal.decompose(y, state['past_idx'], ev, observed=o)
            eff, sup = seasonal.fit_effects(dec)
            fade = min(1.0, sup['holidays'] / float(MIN_HOLIDAYS))
            state['log_adj'][d] = seasonal.future_log_adjustment(state['fut_idx'], ev, eff, scale=EVENT_SCALE * fade)
            state['past_log_adj'][d] = seasonal.past_log_adjustment(state['past_idx'], ev, eff, scale=EVENT_SCALE * fade)
            state['dowf'][d] = dec['dowf'].reindex(state['fut_idx'].dayofweek).values
            sp = seasonal.specialist(dec, state['fut_idx'], state['log_adj'][d], LEVEL_WINDOW)
            if sp is not None and np.isfinite(sp).all():
                state['spec_log'][d] = sp
            n_rep = 0
            if CLEAN_CONTEXT:
                cleaned, n_rep = seasonal.clean_history(y, dec)
                if np.isfinite(cleaned).all():
                    state['clean_history'][d] = cleaned
            diag.append({'target': d, 'holiday_support': sup['holidays'], 'repaired_days': n_rep, 'generic_weekday_effect': round(eff['gen', False], 4), 'generic_weekend_effect': round(eff['gen', True], 4)})
        except Exception as exc:
            state['log_adj'][d] = 0.0
            diag.append({'target': d, 'error': repr(exc)})
    state['event_diagnostics'] = diag
    return (prepared, state)

def select_context(variables, card, state):
    C = min(variables['cutoff_index'], card['limits']['max_context'])
    H = variables['horizon']
    po = variables['past_features']
    pf = variables['known_features']
    known_names = list(variables['known_names'])
    if CALENDAR_COVARIATE and state['daily']:
        factor = np.exp(np.concatenate([state['past_log_adj'], state['log_adj']], axis=1))
        factor = factor.mean(axis=0, keepdims=True)[:, -(C + H):]
        pf = np.concatenate([pf, factor], axis=0) if len(pf) else factor
        known_names = known_names + ['calendar_event_factor']
        state['covariate'] = True
    return ([{'target_indices': list(range(state['N'])), 'targets': np.asarray(state['clean_history'], float)[:, -C:], 'past_only': po[:, -C:] if len(po) else None, 'known_future': pf[:, -(C + H):] if len(pf) else None, 'past_names': variables['past_names'], 'known_names': known_names, 'provenance': 'Event-repaired context (holiday/bridge/eve anomalies replaced by their counterfactual ordinary-day level); calendar effects re-applied after the model'}], state)

def _combine(point, quant, state):
    log_target = state['log_adj']
    lp = np.log(np.maximum(point, 1e-09))
    if GAP_CREDIT:
        base = lp - state['dowf']
        import pandas as _pd
        sm = np.vstack([_pd.Series(row).rolling(9, center=True, min_periods=3).median().values for row in base])
        resid = base - sm
        ev = np.abs(log_target) > 1e-09
        credit = np.zeros_like(log_target)
        ratio = np.clip(np.where(ev, resid / np.where(ev, log_target, 1.0), 0.0), 0.0, 1.0)
        credit[ev] = (ratio * log_target)[ev]
        log_target = log_target - GAP_CREDIT * credit
    mult = np.exp(log_target)
    adjusted = np.maximum(point, 1e-09) * mult
    log_adj = np.log(adjusted)
    sp = state['spec_log']
    use = np.isfinite(sp).all(axis=1) & np.isfinite(log_adj).all(axis=1)
    w = np.where(use, SPECIALIST_WEIGHT, 0.0)[:, None]
    log_final = (1.0 - w) * log_adj + w * np.where(np.isfinite(sp), sp, log_adj)
    new_point = np.exp(log_final) * POINT_UPLIFT
    med = np.maximum(quant[:, :, 4:5], 1e-09)
    off = np.log(np.maximum(quant, 1e-09) / med)
    shaped = med * np.exp(np.where(off < 0, off * LOWER_SHRINK, off * UPPER_SHRINK))
    ratio = (new_point / np.maximum(point, 1e-09))[:, :, None]
    new_quant = np.sort(shaped * ratio, axis=-1)
    return (new_point, new_quant)

def postprocess(outputs, cases, state):
    N, H = (state['N'], state['H'])
    point = np.empty((N, H))
    quantiles = np.empty((N, H, 9))
    for output, case in zip(outputs, cases):
        point[case['target_indices']] = output['point']
        quantiles[case['target_indices']] = output['quantiles']
    point, quantiles = _combine(point, quantiles, state)
    if not np.isfinite(point).all() or not np.isfinite(quantiles).all():
        raise ValueError('non-finite forecast produced by calendar postprocessing')
    return {'point': point, 'quantiles': quantiles, 'components': {'event_log_adjustment': state['log_adj'].tolist(), 'specialist_log_forecast': np.where(np.isfinite(state['spec_log']), state['spec_log'], 0.0).tolist()}}

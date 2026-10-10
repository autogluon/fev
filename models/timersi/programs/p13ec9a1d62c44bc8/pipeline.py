import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from .features import calendar_parts, ewma_causal
    from .specialist import covariate_estimate, absolute_notrend, deviation
    from .combine import blend_distribution, lead_ramp
except Exception:
    from features import calendar_parts, ewma_causal
    from specialist import covariate_estimate, absolute_notrend, deviation
    from combine import blend_distribution, lead_ramp
MAX_OOS_TAIL = 8760
SCALAR_EXTRA = ['cdd65', 'hdd60', 'cdd75', 'temp_ewma48', 'temp_ewma168', 'nonwork']
W_NEAR, W_FAR, RAMP_H = (0.3, 0.55, 120.0)
DEV_SHARE = 0.25
BAND_WIDTH = 1.0

def preprocess(view, card):
    state = {'N': len(view['target_ids']), 'H': view['horizon'], 'C0': view['cutoff_index'], 'item': view.get('item_id'), 'notes': [], 'spec_future': None}
    return (view, state)

def engineer(prepared, card, state):
    v = dict(prepared)
    ts = np.array(v['timestamps'], dtype='datetime64[h]')
    L, H = (v['cutoff_index'], v['horizon'])
    names = list(v['known_names'])
    known = np.asarray(v['known_features'], float) if len(names) else np.zeros((0, L + H))
    Y = np.asarray(v['target_history'], float)
    v['target_history'] = Y
    if 'airtemperature' not in names:
        state['notes'].append('no airtemperature channel; engineer() is a no-op')
        return (v, state)
    T = known[names.index('airtemperature')].astype(float)
    good = np.isfinite(T)
    T = np.where(good, T, np.median(T[good]) if good.any() else 0.0)
    _, dow, is_hol, _, _ = calendar_parts(ts)
    nonwork = np.maximum((dow >= 5).astype(float), is_hol)
    cols = {'cdd65': np.maximum(T - 65.0, 0.0), 'hdd60': np.maximum(60.0 - T, 0.0), 'cdd75': np.maximum(T - 75.0, 0.0), 'temp_ewma48': ewma_causal(T, 48), 'temp_ewma168': ewma_causal(T, 168), 'nonwork': nonwork}
    est = np.zeros((state['N'], L + H))
    for d in range(state['N']):
        y = Y[d]
        fill = float(np.nanmean(y[np.isfinite(y)])) if np.isfinite(y).any() else 0.0
        try:
            fut, oos = covariate_estimate(ts, T, y, H, n_blocks=4, oos_tail=MAX_OOS_TAIL)
        except Exception as exc:
            state['notes'].append('specialist failed: %r' % (exc,))
            est[d, :L] = np.where(np.isfinite(y), y, fill)
            est[d, L:] = fill
            continue
        past = oos if oos is not None else np.full(L, np.nan)
        past = np.where(np.isfinite(past), past, np.where(np.isfinite(y), y, fill))
        est[d, :L] = past
        est[d, L:] = np.where(np.isfinite(fut), fut, fill)
    state['spec_oos'] = est[:, :L].copy()
    combo = np.empty((state['N'], H))
    for d in range(state['N']):
        y = Y[d]
        base = est[d, L:]
        try:
            absolute = absolute_notrend(ts, T, y, H)
        except Exception as exc:
            state['notes'].append('absolute_notrend failed: %r' % (exc,))
            absolute = None
        if absolute is None or not np.isfinite(absolute).all():
            absolute = base
        try:
            dev = deviation(ts, T, y, H)
        except Exception as exc:
            state['notes'].append('deviation failed: %r' % (exc,))
            dev = None
        if dev is None or not np.isfinite(dev).all():
            combo[d] = absolute
            state['notes'].append('deviation unavailable for target %d' % d)
        else:
            combo[d] = (1.0 - DEV_SHARE) * absolute + DEV_SHARE * dev
    state['spec_future'] = combo
    est_names = ['load_estimate'] if state['N'] == 1 else ['load_estimate_%d' % d for d in range(state['N'])]
    v['known_names'] = names + SCALAR_EXTRA + est_names
    v['known_features'] = np.vstack([known] + [cols[n][None, :] for n in SCALAR_EXTRA] + [est])
    state['notes'].append('known covariates %d -> %d' % (len(names), len(v['known_names'])))
    return (v, state)

def select_context(variables, card, state):
    C = min(variables['cutoff_index'], card['limits']['max_context'])
    H = variables['horizon']
    po = np.asarray(variables['past_features'], float) if len(variables['past_names']) else None
    pf = np.asarray(variables['known_features'], float)
    tgt = np.asarray(variables['target_history'], float)[:, -C:]
    kf = pf[:, -(C + H):] if pf.size else None
    if kf is not None:
        kf = np.where(np.isfinite(kf), kf, 0.0)
    return ([{'target_indices': list(range(state['N'])), 'targets': np.where(np.isfinite(tgt), tgt, 0.0), 'past_only': po[:, -C:] if po is not None and po.size else None, 'known_future': kf, 'past_names': list(variables['past_names']), 'known_names': list(variables['known_names']), 'provenance': 'All targets, max context, + causal degree-day/EWMA/operating-state/load-estimate known covariates'}], state)

def postprocess(outputs, cases, state):
    N, H = (state['N'], state['H'])
    point = np.empty((N, H))
    quantiles = np.empty((N, H, 9))
    for output, case in zip(outputs, cases):
        point[case['target_indices']] = np.asarray(output['point'], float)
        quantiles[case['target_indices']] = np.asarray(output['quantiles'], float)
    spec = state.get('spec_future')
    if spec is None or np.asarray(spec).shape != (N, H) or (not np.isfinite(spec).all()):
        state['notes'].append('no usable specialist path; native distribution returned as is')
        return {'point': point, 'quantiles': np.sort(quantiles, axis=-1)}
    w = lead_ramp(H, W_NEAR, W_FAR, RAMP_H)
    point, quantiles = blend_distribution(point, quantiles, spec, w, BAND_WIDTH)
    return {'point': point, 'quantiles': quantiles, 'components': {'specialist_weight_near': W_NEAR, 'specialist_weight_far': W_FAR, 'ramp_hours': RAMP_H, 'deviation_share': DEV_SHARE, 'band_width': BAND_WIDTH}}

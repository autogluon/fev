import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import expert as EX
except ImportError:
    from . import expert as EX
CONFIG = {'cal_cov': True, 'hdd_cdd': True, 'expert': {'enabled': True, 'w_special': 0.9, 'w_flip': 0.7, 'w_base': 0.45, 'ramp': (4, 28), 'min_len': 5 * EX.WEEK + 96}}

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

def _day_class_gate(ts, L, H):
    cal = EX.calendar_frame(ts)
    fut = slice(L, L + H)
    last = slice(max(0, L - 96), L)
    hol = EX.holiday_flag(cal)
    xw = EX.xmas_window(cal)
    special = bool(hol[fut].max() > 0 or xw[fut].max() > 0)

    def cls(sl):
        dow = int(np.bincount(cal['dow'][sl] if np.ndim(cal['dow'][sl]) else [0]).argmax())
        off = bool(hol[sl].max() > 0 or xw[sl].max() > 0)
        return 2 if off or dow == 6 else 1 if dow == 5 else 0
    flip = cls(fut) != cls(last)
    return (special, flip)

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
        rows, nms = _calendar_covariates(ts)
        extra.append(rows)
        extra_names += nms
    if cfg['hdd_cdd'] and 'temperature' in names:
        t = known[names.index('temperature')]
        extra.append(np.stack([np.maximum(16.0 - t, 0.0), np.maximum(t - 20.0, 0.0)]))
        extra_names += ['temp_hdd16', 'temp_cdd20']
    if extra:
        allrows = np.concatenate([known] + extra, axis=0) if len(known) else np.concatenate(extra, axis=0)
        v['known_features'] = np.nan_to_num(allrows, nan=0.0, posinf=0.0, neginf=0.0)
        v['known_names'] = names + extra_names
    state['components']['constructed_known_covariates'] = extra_names
    state['exp_inputs'] = {'ts': ts, 'y': np.asarray(v['target_history'], dtype=np.float64), 'known_orig': known, 'L': int(v['cutoff_index'])}
    return (v, state)

def select_context(variables, card, state):
    L = int(variables['cutoff_index'])
    H = int(variables['horizon'])
    C = min(L, card['limits']['max_context'])
    po = np.asarray(variables['past_features'])
    pf = np.asarray(variables['known_features'])
    tg = np.asarray(variables['target_history'])
    cases = [{'target_indices': list(range(state['N'])), 'targets': tg[:, L - C:L], 'past_only': po[:, L - C:L] if len(po) else None, 'known_future': pf[:, L - C:L + H] if len(pf) else None, 'past_names': variables['past_names'], 'known_names': variables['known_names'], 'provenance': 'primary, official + constructed semantic covariates, C=%d' % C}]
    return (cases, state)

def _train_expert(state):
    xi = state['exp_inputs']
    y, ts, known, L = (xi['y'], xi['ts'], xi['known_orig'], xi['L'])
    H = state['H']
    cfg = state['cfg']['expert']
    if L < cfg['min_len']:
        state['notes'].append('expert skipped: history %d < %d' % (L, cfg['min_len']))
        return None
    preds = []
    for d in range(y.shape[0]):
        pr, model, names = EX.fit_predict(y[d], ts, known if known.size else None, L, H=H)
        preds.append(pr)
        if d == 0:
            state['components']['expert_fit'] = {'n_features': int(model.n_features_in_), 'n_estimators': int(model.n_estimators), 'features': names}
    return np.stack(preds)

def postprocess(outputs, cases, state):
    N, H = (state['N'], state['H'])
    point = np.empty((N, H))
    quantiles = np.empty((N, H, 9))
    for output, case in zip(outputs, cases):
        point[case['target_indices']] = output['point']
        quantiles[case['target_indices']] = output['quantiles']
    cfg = state['cfg']['expert']
    comp = state['components']
    comp['expert_active'] = False
    if cfg['enabled']:
        try:
            E = _train_expert(state)
        except Exception as exc:
            E = None
            state['notes'].append('expert failed: %r' % (exc,))
        if E is not None:
            xi = state['exp_inputs']
            special, flip = _day_class_gate(xi['ts'], xi['L'], H)
            wmax = cfg['w_special'] if special else cfg['w_flip'] if flip else cfg['w_base']
            h0, h1 = cfg['ramp']
            w = wmax * np.clip((np.arange(H) - h0) / float(h1 - h0), 0.0, 1.0)
            shift = w[None, :] * (E - quantiles[:, :, 4])
            point = point + shift
            quantiles = quantiles + shift[:, :, None]
            comp.update(expert_active=True, day_special=bool(special), day_flip=bool(flip), wmax=float(wmax), ramp=list(cfg['ramp']), expert_rel_shift_per_item=[float(x) for x in np.mean(shift, axis=1) / np.maximum(np.abs(point).mean(axis=1), 1e-09)])
    quantiles = np.sort(quantiles, axis=2)
    return {'point': point, 'quantiles': quantiles, 'components': dict(comp, notes=state['notes'])}

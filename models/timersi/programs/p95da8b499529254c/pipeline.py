import numpy as np
import calendar_features as cf
import specialist
BLEND_W0, BLEND_W1 = (0.45, 0.85)
SHIFT_CAP = 0.25
SPECIALIST_CONFIGS = ((2.0, 250), (2.0, 400), (3.0, 400), (4.5, 400))

def _unix(timestamps):
    return np.asarray(timestamps, dtype='datetime64[s]').astype(np.int64)

def preprocess(view, card):
    hist = np.asarray(view['target_history'], dtype=np.float64)
    obs = np.asarray(view['target_observed']) if len(view.get('target_observed', [])) else None
    state = {'N': hist.shape[0], 'H': int(view['horizon']), 'L': int(view['cutoff_index']), 'item_id': view.get('item_id'), 'unix': _unix(view['timestamps']), 'history': hist, 'observed': obs}
    return (view, state)

def engineer(prepared, card, state):
    names = list(prepared['known_names'])
    known = np.asarray(prepared['known_features'], dtype=np.float64)
    temp = known[names.index('temperature')] if 'temperature' in names else None
    cal_names, cal = cf.build(state['item_id'], state['unix'], temp)
    known = np.concatenate([known, cal], axis=0) if known.size else cal
    out = dict(prepared)
    out['known_features'] = known
    out['known_names'] = names + cal_names
    state['known'] = known
    state['known_names'] = names + cal_names
    return (out, state)

def select_context(variables, card, state):
    C = min(variables['cutoff_index'], card['limits']['max_context'])
    H = variables['horizon']
    po = np.asarray(variables['past_features'])
    pf = np.asarray(variables['known_features'])
    return ([{'target_indices': list(range(state['N'])), 'targets': np.asarray(variables['target_history'])[:, -C:], 'past_only': po[:, -C:] if po.size else None, 'known_future': pf[:, -(C + H):] if pf.size else None, 'past_names': list(variables['past_names']), 'known_names': list(variables['known_names']), 'provenance': 'Reference context 15360 + causal calendar/day-type/HDD-CDD known columns'}], state)

def _column(state, name):
    try:
        return state['known'][state['known_names'].index(name)]
    except (ValueError, KeyError):
        return None

def specialist_shapes(state):
    L, H = (state['L'], state['H'])
    temp = _column(state, 'temperature')
    rdif = _column(state, 'radiation_diffuse_horizontal')
    rdir = _column(state, 'radiation_direct_horizontal')
    if temp is None or rdif is None or rdir is None:
        return None
    unix = state['unix']
    if len(unix) < L + H:
        return None
    out = np.zeros((state['N'], H))
    ok = np.zeros(state['N'], dtype=bool)
    for d in range(state['N']):
        hist = state['history'][d]
        if state['observed'] is not None and (not np.all(state['observed'][d])):
            continue
        if not np.isfinite(hist).all():
            continue
        acc, n = (np.zeros(H), 0)
        for years, trees in SPECIALIST_CONFIGS:
            try:
                f = specialist.forecast(state['item_id'], unix[:L + H], hist, temp[:L + H], rdif[:L + H], rdir[:L + H], H, years=years, n_estimators=trees)
            except Exception:
                f = None
            if f is None:
                continue
            acc += f - f.mean()
            n += 1
        if n == 0:
            continue
        out[d] = acc / n
        ok[d] = True
    return (out, ok)

def postprocess(outputs, cases, state):
    point = np.empty((state['N'], state['H']))
    quantiles = np.empty((state['N'], state['H'], 9))
    for output, case in zip(outputs, cases):
        point[case['target_indices']] = np.asarray(output['point'])
        quantiles[case['target_indices']] = np.asarray(output['quantiles'])
    try:
        res = specialist_shapes(state)
    except Exception:
        res = None
    if res is not None:
        shape, ok = res
        native_shape = point - point.mean(axis=1, keepdims=True)
        ramp = BLEND_W0 + (BLEND_W1 - BLEND_W0) * np.arange(state['H']) / max(1, state['H'] - 1)
        shift = ramp[None, :] * (shape - native_shape)
        cap = SHIFT_CAP * np.abs(point.mean(axis=1, keepdims=True))
        shift = np.clip(shift, -cap, cap)
        shift[~ok] = 0.0
        point = point + shift
        quantiles = quantiles + shift[:, :, None]
    quantiles = np.sort(quantiles, axis=2)
    return {'point': point, 'quantiles': quantiles, 'components': {'native_point': point - (shift if res is not None else 0.0)}}

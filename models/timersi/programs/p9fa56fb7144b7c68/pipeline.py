import numpy as np
W_NEAR = 0.2
W_FAR = 0.6
RHO = 0.6
SEEDS = (0, 1, 2)

def _blend_weights(H):
    h = np.arange(H) / max(H - 1, 1)
    return W_NEAR + (W_FAR - W_NEAR) * h

def _spread_factor(w):
    return np.sqrt(np.maximum(1.0 - 2.0 * w * (1.0 - w) * (1.0 - RHO), 1e-06))

def _asarray(x, dtype=None):
    return np.asarray(x, dtype=dtype)

def preprocess(view, card):
    state = {'N': len(view['target_ids']), 'H': view['horizon'], 'view': view, 'card': card}
    return (view, state)
CAL_HARMONICS = ((168, 1), (168, 2), (24, 1), (24, 2), (24, 3))

def engineer(prepared, card, state):
    view = dict(prepared)
    ts = np.asarray(view['timestamps']).astype('datetime64[h]')
    hours = ((ts - np.datetime64('1970-01-01T00')) / np.timedelta64(1, 'h')).astype(np.float64)
    extra, names = ([], [])
    for period, n in CAL_HARMONICS:
        extra.append(np.sin(2 * np.pi * n * hours / period))
        names.append(f'cal_sin_{period}_{n}')
        extra.append(np.cos(2 * np.pi * n * hours / period))
        names.append(f'cal_cos_{period}_{n}')
    dow = (hours // 24 + 4) % 7
    extra.append((dow >= 5).astype(np.float64))
    names.append('cal_weekend')
    kf = np.asarray(view['known_features'], float)
    kf = np.vstack([kf, np.asarray(extra)]) if kf.size else np.asarray(extra)
    view['known_features'] = kf
    view['known_names'] = np.concatenate([np.asarray(view['known_names'], dtype=object), np.asarray(names, dtype=object)]).astype(str)
    state['view'] = prepared
    return (view, state)
SEASON_WEEK = 168
PATCH = 32

def _aligned_context(L, cap):
    period = SEASON_WEEK * PATCH // np.gcd(SEASON_WEEK, PATCH)
    C = min(L, cap)
    C = C // period * period
    return int(C) if C >= period else int(min(L, cap))

def select_context(variables, card, state):
    C = min(variables['cutoff_index'], card['limits']['max_context'])
    H = variables['horizon']
    po = variables['past_features']
    pf = variables['known_features']
    return ([{'target_indices': list(range(state['N'])), 'targets': variables['target_history'][:, -C:], 'past_only': po[:, -C:] if len(po) else None, 'known_future': pf[:, -(C + H):] if len(pf) else None, 'past_names': variables['past_names'], 'known_names': variables['known_names'], 'provenance': 'Official reference inputs, all targets/covariates, max_context=15360'}], state)

def _local_forecast(view, H):
    try:
        from localmodel import fit_predict
    except ImportError:
        import os
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from localmodel import fit_predict
    hist = _asarray(view['target_history'], float)
    known = _asarray(view['known_features'], float)
    ts = _asarray(view['timestamps'])
    if known.ndim != 2 or known.shape[0] < 1:
        return None
    L = int(view['cutoff_index'])
    if L < 12000:
        return None
    out = np.empty((hist.shape[0], H))
    for d in range(hist.shape[0]):
        r = fit_predict(hist[d], known, ts, H, backtest_origins=0, seeds=SEEDS)
        out[d] = r['point']
    return out

def postprocess(outputs, cases, state):
    N, H = (state['N'], state['H'])
    point = np.empty((N, H))
    quantiles = np.empty((N, H, 9))
    for output, case in zip(outputs, cases):
        point[case['target_indices']] = output['point']
        quantiles[case['target_indices']] = output['quantiles']
    try:
        local = _local_forecast(state['view'], H)
    except Exception:
        local = None
    components = {'native_point': point.copy()}
    if local is not None and np.all(np.isfinite(local)):
        components['local_point'] = local
        w = _blend_weights(H)[None, :]
        blended = (1.0 - w) * point + w * local
        quantiles = quantiles + (blended - point)[:, :, None]
        s = _spread_factor(w[0])[None, :, None]
        med = quantiles[:, :, 4:5]
        quantiles = med + s * (quantiles - med)
        point = blended
    quantiles = np.sort(quantiles, axis=2)
    return {'point': point, 'quantiles': quantiles, 'components': components}

import numpy as np
from calendar_tools import decode
import counter
W_MAX, W_MIN, SPREAD_A = (0.97, 0.2, 1.0)
GUIDE_BOOST = 2.0
HALF_LIVES = (4.0, 8.0, 16.0)
NSIM = 500

def preprocess(view, card):
    hour, dow, days = decode(view['timestamps'])
    state = {'N': len(view['target_ids']), 'H': view['horizon'], 'L': int(view['cutoff_index']), 'hour': hour, 'dow': dow, 'days': days}
    return (view, state)

def engineer(prepared, card, state):
    state['history'] = np.asarray(prepared['target_history'], dtype=float)
    state['observed'] = prepared.get('target_observed')
    return (prepared, state)

def select_context(variables, card, state):
    C = min(variables['cutoff_index'], card['limits']['max_context'])
    H = variables['horizon']
    po = variables['past_features']
    pf = variables['known_features']
    return ([{'target_indices': list(range(state['N'])), 'targets': variables['target_history'][:, -C:], 'past_only': po[:, -C:] if len(po) else None, 'known_future': pf[:, -(C + H):] if len(pf) else None, 'past_names': variables['past_names'], 'known_names': variables['known_names'], 'provenance': 'Reference inputs (probe 1 rejected regrouping), max_context=15360'}], state)

def _specialist(state, native_point=None):
    L, H = (state['L'], state['H'])
    hour, dow, days = (state['hour'], state['dow'], state['days'])
    ph, pd_, pday = (hour[:L], dow[:L], days[:L])
    fh, fd = (hour[L:L + H], dow[L:L + H])
    hist = state['history']
    N = hist.shape[0]
    Q = np.empty((N, H, 9))
    ok = np.zeros(N, dtype=bool)
    for i in range(N):
        x = hist[i]
        if not np.all(np.isfinite(x)):
            continue
        guide = -1
        if native_point is not None:
            try:
                guide = counter.native_drop_hour(native_point[i], float(x[-1]))
            except Exception:
                guide = -1
        try:
            _, q, _ = counter.forecast_guided(x, ph, pd_, pday, fh, fd, guide=guide, boost=GUIDE_BOOST, nsim=NSIM, seed=i, half_lives=HALF_LIVES)
        except Exception:
            continue
        if q.shape == (H, 9) and np.all(np.isfinite(q)):
            Q[i] = np.sort(q, axis=1)
            ok[i] = True
    return (Q, ok)

def postprocess(outputs, cases, state):
    N, H = (state['N'], state['H'])
    point = np.empty((N, H))
    quantiles = np.empty((N, H, 9))
    for output, case in zip(outputs, cases):
        point[case['target_indices']] = output['point']
        quantiles[case['target_indices']] = output['quantiles']
    Qs, ok = _specialist(state, native_point=point.copy())
    if ok.any():
        hist = state['history']
        m = min(24, max(1, hist.shape[1] // 2))
        scale = np.abs(hist[:, m:] - hist[:, :-m]).mean(axis=1)
        scale = np.where(np.isfinite(scale) & (scale > 1e-08), scale, 1.0)
        spread = (Qs[:, :, 7] - Qs[:, :, 1]) / scale[:, None]
        w = np.clip(W_MAX / (1.0 + SPREAD_A * spread), W_MIN, W_MAX)
        w = np.where(ok[:, None], w, 0.0)[:, :, None]
        quantiles = np.sort((1.0 - w) * quantiles + w * Qs, axis=2)
        point = np.where(ok[:, None], quantiles[:, :, 4], point)
        state['mean_w'] = float(w.mean())
    state['spec_frac'] = float(ok.mean())
    return {'point': point, 'quantiles': quantiles}

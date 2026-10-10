import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import solar_calendar as sc
ALPHA = 0.78
ALPHA_TAU = 0.4
ALPHA_CENTER = 0.94
ALPHA_SPAN = 0.12
ALPHA_LO, ALPHA_HI = (0.35, 1.15)
EXTRAP_LAMBDA = 3.6
W_MAX = 0.6
MIN_SWING = 0.02
KAPPA = 1.05
R_MIN = 0.65
Z90 = 1.2815515655446004

def _index_for(ts, L, alpha):
    S = sc.insolation_index(ts, alpha=float(alpha))
    if not np.all(np.isfinite(S)) or np.min(S) <= 0:
        return None
    S = S / float(np.mean(S[:L]))
    if np.max(S) - np.min(S) < MIN_SWING:
        return None
    ls = np.log(S)
    reach = float(np.max(np.abs(ls[L:]))) if ls.size > L else 0.0
    gamma = 1.0 / (1.0 + reach / EXTRAP_LAMBDA)
    return np.exp(ls * gamma)

def _fit_alpha(logx, y, obs):
    m = obs & np.isfinite(y) & (y > 0)
    if m.sum() < 6:
        return (ALPHA, 0.0)
    lx = logx[m] - logx[m].mean()
    V = float(lx @ lx)
    if V <= 1e-08:
        return (ALPHA, 0.0)
    ly = np.log(y[m])
    ahat = float(lx @ (ly - ly.mean()) / V)
    w = min(W_MAX, V / (V + ALPHA_TAU))
    dev = float(np.clip(w * (ahat - ALPHA_CENTER), -ALPHA_SPAN, ALPHA_SPAN))
    return (float(np.clip(ALPHA + dev, ALPHA_LO, ALPHA_HI)), w)

def preprocess(view, card):
    L = int(view['cutoff_index'])
    H = int(view['horizon'])
    state = {'N': len(view['target_ids']), 'H': H, 'L': L}
    ts = list(view['timestamps'])
    hist = np.asarray(view['target_history'], float)
    obs = np.asarray(view.get('target_observed', np.isfinite(hist)), bool)
    use = np.zeros(state['N'], bool)
    S = [None] * state['N']
    if len(ts) >= L + H:
        ts = ts[:L + H]
        logx = np.log(sc.h0_week_sum(ts))[:L]
        for i in range(state['N']):
            y = hist[i]
            m = obs[i] & np.isfinite(y)
            if m.sum() < 4 or np.nanmin(y[m]) <= 0:
                continue
            a_i, _ = _fit_alpha(logx, y, m)
            Si = _index_for(ts, L, a_i)
            if Si is None:
                continue
            S[i] = Si
            use[i] = True
    state['S'] = S
    state['use'] = use
    return (view, state)

def _own_past_log_spread(d, drop_first):
    v = d[1:] if drop_first and d.size > 6 else d
    v = v[np.isfinite(v) & (v > 0)]
    if v.size < 6:
        return None
    lv = np.log(v)
    dev = lv - np.median(lv)
    return float(np.percentile(dev, 90) - np.percentile(dev, 10)) / (2.0 * Z90)

def engineer(prepared, card, state):
    if not state['use'].any():
        return (prepared, state)
    L = state['L']
    S = state['S']
    hist = np.array(prepared['target_history'], dtype=float)
    out = hist.copy()
    use = state['use']
    for i in np.flatnonzero(use):
        out[i] = hist[i] / S[i][:L]
    obs = np.asarray(prepared.get('target_observed', np.isfinite(hist)), bool)
    sig = np.full(state['N'], np.nan)
    for i in np.flatnonzero(use):
        v = out[i][obs[i] & np.isfinite(out[i])]
        s_i = _own_past_log_spread(v, drop_first=True)
        if s_i is not None:
            sig[i] = s_i
    state['sigma_past'] = sig
    variables = dict(prepared)
    variables['target_history'] = out
    return (variables, state)

def select_context(variables, card, state):
    C = min(variables['cutoff_index'], card['limits']['max_context'])
    H = variables['horizon']
    po = variables['past_features']
    pf = variables['known_features']
    return ([{'target_indices': list(range(state['N'])), 'targets': np.asarray(variables['target_history'], float)[:, -C:], 'past_only': po[:, -C:] if len(po) else None, 'known_future': pf[:, -(C + H):] if len(pf) else None, 'past_names': variables['past_names'], 'known_names': variables['known_names'], 'provenance': 'Solar-geometry deseasonalised targets, full context'}], state)

def postprocess(outputs, cases, state):
    N, H = (state['N'], state['H'])
    point = np.empty((N, H))
    quantiles = np.empty((N, H, 9))
    for output, case in zip(outputs, cases):
        point[case['target_indices']] = output['point']
        quantiles[case['target_indices']] = output['quantiles']
    if state['use'].any():
        L = state['L']
        for i in np.flatnonzero(state['use']):
            f = state['S'][i][L:L + H]
            point[i] = np.maximum(point[i] * f, 0.0)
            quantiles[i] = np.maximum(quantiles[i] * f[:, None], 0.0)
        quantiles = _calibrate_spread(quantiles, state)
    quantiles = np.sort(quantiles, axis=2)
    return {'point': point, 'quantiles': quantiles}

def _calibrate_spread(quantiles, state):
    sig = state.get('sigma_past')
    if sig is None:
        return quantiles
    q = quantiles.copy()
    for i in np.flatnonzero(state['use']):
        if not np.isfinite(sig[i]) or sig[i] <= 0:
            continue
        lo, hi, med = (q[i, :, 0], q[i, :, 8], q[i, :, 4])
        ok = (lo > 0) & (med > 0) & (hi > lo)
        if not ok.any():
            continue
        sig_n = np.where(ok, np.log(np.maximum(hi, 1e-12) / np.maximum(lo, 1e-12)) / (2.0 * Z90), np.inf)
        r = np.clip(KAPPA * sig[i] / np.maximum(sig_n, 1e-09), R_MIN, 1.0)
        r = np.where(ok, r, 1.0)
        base = np.maximum(med, 1e-12)[:, None]
        q[i] = np.where(ok[:, None], base * np.exp(np.log(np.maximum(q[i], 1e-12) / base) * r[:, None]), q[i])
    return q

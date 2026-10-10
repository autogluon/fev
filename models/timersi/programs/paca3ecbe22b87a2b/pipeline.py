import numpy as np
Z9 = np.array([-1.2815515655446004, -0.8416212335729143, -0.5244005127080407, -0.2533471031357997, 0.0, 0.2533471031357997, 0.5244005127080407, 0.8416212335729143, 1.2815515655446004])
SIGMA8 = 0.6
SIG_POW = 0.5
INC_FLOOR = 0.5

def _observed_tail(hist, obs):
    hist = np.asarray(hist, dtype=float)
    obs = np.asarray(obs, dtype=float) > 0
    idx = np.flatnonzero(obs & np.isfinite(hist))
    if idx.size == 0:
        good = np.flatnonzero(np.isfinite(hist))
        return hist[good] if good.size else np.array([0.0])
    return hist[idx[0]:idx[-1] + 1]

def preprocess(view, card):
    hist = np.asarray(view['target_history'], dtype=float)
    obs = np.asarray(view['target_observed'], dtype=float)
    N, H = (hist.shape[0], int(view['horizon']))
    last = np.zeros(N)
    scale = np.ones(N)
    for d in range(N):
        tail = _observed_tail(hist[d], obs[d])
        tail = tail[np.isfinite(tail)]
        if tail.size == 0:
            tail = np.array([0.0])
        last[d] = tail[-1]
        if tail.size >= 2:
            dd = np.abs(np.diff(tail))
            scale[d] = float(dd.mean()) if dd.mean() > 0 else 1.0
    state = {'N': N, 'H': H, 'last': last, 'scale': scale, 'item': view.get('item_id', '')}
    return (view, state)

def engineer(prepared, card, state):
    return (prepared, state)

def select_context(variables, card, state):
    C = min(int(variables['cutoff_index']), int(card['limits']['max_context']))
    H = int(variables['horizon'])
    po = np.asarray(variables['past_features'], dtype=float)
    pf = np.asarray(variables['known_features'], dtype=float)
    return ([{'target_indices': list(range(state['N'])), 'targets': np.asarray(variables['target_history'], dtype=float)[:, -C:], 'past_only': po[:, -C:] if po.size else None, 'known_future': pf[:, -(C + H):] if pf.size else None, 'past_names': variables['past_names'], 'known_names': variables['known_names'], 'provenance': 'P1: reference inputs unchanged (all targets/covariates, max_context=15360); only the output geometry is corrected.'}], state)

def postprocess(outputs, cases, state):
    N, H = (state['N'], state['H'])
    last, scale = (state['last'], state['scale'])
    point = np.empty((N, H))
    quant = np.empty((N, H, 9))
    for output, case in zip(outputs, cases):
        point[case['target_indices']] = np.asarray(output['point'], dtype=float)
        quant[case['target_indices']] = np.asarray(output['quantiles'], dtype=float)
    point = np.where(np.isfinite(point), point, last[:, None])
    quant = np.where(np.isfinite(quant), quant, point[:, :, None])
    point = np.maximum.accumulate(np.maximum(point, last[:, None]), axis=1)
    hh = np.arange(1, H + 1, dtype=float)
    sig = SIGMA8 * (hh / float(H)) ** SIG_POW
    qs = np.sort(quant, axis=-1)
    dev = qs - qs[:, :, 4:5]
    inc = np.maximum(point - last[:, None], INC_FLOOR * scale[:, None] * hh[None, :])
    inc = np.maximum(inc, 1e-09)
    sdev = inc[:, :, None] * (np.exp(Z9[None, None, :] * sig[None, :, None]) - 1.0)
    union = np.where(Z9[None, None, :] < 0, np.minimum(dev, sdev), np.maximum(dev, sdev))
    union[:, :, 4] = 0.0
    quant = point[:, :, None] + union
    quant = np.sort(quant, axis=-1)
    quant = np.maximum.accumulate(np.maximum(quant, last[:, None, None]), axis=1)
    point = np.clip(point, quant[:, :, 0], quant[:, :, 8])
    return {'point': point, 'quantiles': quant}

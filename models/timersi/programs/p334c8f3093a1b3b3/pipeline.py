import numpy as np
SNAP_PREFIX = 'snap_'

def preprocess(view, card):
    state = {'N': len(view['target_ids']), 'H': view['horizon'], 'item': view.get('item_id'), 'notes': {}, 'history': np.asarray(view['target_history'], dtype=float), 'observed': np.asarray(view['target_observed'], dtype=bool)}
    return (view, state)
SCREEN_ALPHA = 0.05

def _keep_mask(names, feats, static, target, cutoff):
    keep = np.ones(len(names), dtype=bool)
    if len(names) == 0:
        return keep
    arr = np.asarray(feats, dtype=float)
    own_state = str(static.get('state_id', '') or '')
    for j, nm in enumerate(names):
        col = arr[j]
        finite = col[np.isfinite(col)]
        if finite.size == 0:
            keep[j] = False
            continue
        if float(np.max(finite) - np.min(finite)) <= 1e-12:
            keep[j] = False
            continue
        if nm.startswith(SNAP_PREFIX) and own_state:
            if nm[len(SNAP_PREFIX):] != own_state:
                keep[j] = False
    m = max(int(keep.sum()), 1)
    for j in np.flatnonzero(keep):
        if not _associated(arr[j][:cutoff], target, SCREEN_ALPHA / m):
            keep[j] = False
    return keep

def _associated(past_col, target, alpha):
    x = np.asarray(past_col, dtype=float)
    y = np.asarray(target, dtype=float)
    n = min(x.size, y.size)
    if n < 6:
        return False
    x, y = (x[-n:], y[-n:])
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = (x[ok], y[ok])
    if x.size < 6:
        return False
    dx, dy = (np.diff(x), np.diff(y))
    if np.ptp(dx) <= 1e-12 or np.ptp(dy) <= 1e-12:
        return False
    rx = _rank(dx)
    ry = _rank(dy)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = np.sqrt(float(np.dot(rx, rx)) * float(np.dot(ry, ry)))
    if den <= 0:
        return False
    rho = float(np.dot(rx, ry)) / den
    nn = dx.size
    if nn <= 2 or abs(rho) >= 1.0:
        return abs(rho) >= 1.0 and nn > 2
    t = abs(rho) * np.sqrt((nn - 2) / max(1.0 - rho * rho, 1e-12))
    p = _t_sf(t, nn - 2) * 2.0
    return p < alpha

def _rank(a):
    order = np.argsort(a, kind='mergesort')
    r = np.empty(a.size, dtype=float)
    r[order] = np.arange(a.size, dtype=float)
    sa = a[order]
    i = 0
    while i < a.size:
        j = i
        while j + 1 < a.size and sa[j + 1] == sa[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = np.mean(r[order[i:j + 1]])
        i = j + 1
    return r

def _t_sf(t, df):
    if df <= 0:
        return 1.0
    x = df / (df + t * t)
    return 0.5 * _betainc(0.5 * df, 0.5, x)

def _betainc(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = _lgamma(a) + _lgamma(b) - _lgamma(a + b)
    front = np.exp(a * np.log(x) + b * np.log(1.0 - x) - lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - np.exp(b * np.log(1.0 - x) + a * np.log(x) - lbeta) * _betacf(b, a, 1.0 - x) / b

def _betacf(a, b, x, itmax=200, eps=3e-12):
    qab, qap, qam = (a + b, a + 1.0, a - 1.0)
    c, d = (1.0, 1.0 - qab * x / qap)
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < 1e-30:
            d = 1e-30
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        if abs(d) < 1e-30:
            d = 1e-30
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h

def _lgamma(z):
    g = [676.5203681218851, -1259.1392167224028, 771.3234287776531, -176.6150291621406, 12.507343278686905, -0.13857109526572012, 9.984369578019572e-06, 1.5056327351493116e-07]
    if z < 0.5:
        return np.log(np.pi / abs(np.sin(np.pi * z))) - _lgamma(1.0 - z)
    z -= 1.0
    x = 0.9999999999998099
    for i, gi in enumerate(g):
        x += gi / (z + i + 1.0)
    t = z + 7.5
    return 0.5 * np.log(2 * np.pi) + (z + 0.5) * np.log(t) - t + np.log(x)

def engineer(prepared, card, state):
    view = prepared
    static = view.get('static', {}) or {}
    tgt = np.asarray(view['target_history'], dtype=float)
    obs = np.asarray(view['target_observed'], dtype=bool)
    ref = tgt[0].copy()
    ref[~obs[0]] = np.nan
    cutoff = int(view['cutoff_index'])
    kn = np.asarray(view['known_features'], dtype=float) if len(view['known_names']) else np.zeros((0, 0))
    kmask = _keep_mask(list(view['known_names']), kn, static, ref, cutoff)
    po = np.asarray(view['past_features'], dtype=float) if len(view['past_names']) else np.zeros((0, 0))
    pmask = _keep_mask(list(view['past_names']), po, static, ref, cutoff)
    state['known_mask'] = kmask
    state['past_mask'] = pmask
    state['notes']['dropped_known'] = [n for n, k in zip(view['known_names'], kmask) if not k]
    return (view, state)

def select_context(variables, card, state):
    C = min(variables['cutoff_index'], card['limits']['max_context'])
    H = variables['horizon']
    km, pm = (state['known_mask'], state['past_mask'])
    po = np.asarray(variables['past_features'], dtype=float) if len(variables['past_names']) else None
    kf = np.asarray(variables['known_features'], dtype=float) if len(variables['known_names']) else None
    past_only = None
    past_names = []
    if po is not None and pm.any():
        past_only = po[pm][:, -C:]
        past_names = [n for n, k in zip(variables['past_names'], pm) if k]
    known_future = None
    known_names = []
    if kf is not None and km.any():
        known_future = kf[km][:, -(C + H):]
        known_names = [n for n, k in zip(variables['known_names'], km) if k]
    dropped = state['notes']['dropped_known']
    return ([{'target_indices': list(range(state['N'])), 'targets': np.asarray(variables['target_history'], dtype=float)[:, -C:], 'past_only': past_only, 'known_future': known_future, 'past_names': past_names, 'known_names': known_names, 'provenance': 'All targets, context=%d, pruned zero-variance / foreign-state covariates: %s' % (C, ','.join(dropped) if dropped else 'none')}], state)
DISP_C = 0.7
_G = 2.563 / 0.798
_SHRINK = 3.0

def _step_dispersion(hist, horizon):
    h = np.asarray(hist, dtype=float)
    h = h[np.isfinite(h)]
    if h.size < 2:
        base = float(abs(h[-1])) if h.size else 1.0
        return np.maximum(np.full(horizon, max(base, 1e-09)) * np.sqrt(np.arange(1, horizon + 1)), 1e-09)
    d1 = max(float(np.mean(np.abs(np.diff(h)))), 1e-09)
    out = np.empty(horizon)
    for k in range(1, horizon + 1):
        prior = d1 * np.sqrt(k)
        if h.size > k:
            dk = np.abs(h[k:] - h[:-k])
            n = float(dk.size)
            w = n / (n + _SHRINK)
            out[k - 1] = w * float(np.mean(dk)) + (1.0 - w) * prior
        else:
            out[k - 1] = prior
    return np.maximum(np.maximum.accumulate(out), 1e-09)

def _widen(point, quant, hist):
    horizon = quant.shape[0]
    target_w = DISP_C * _step_dispersion(hist, horizon) * _G
    native_w = np.maximum(quant[:, 8] - quant[:, 0], 1e-09)
    scale = np.maximum(1.0, target_w / native_w)[:, None]
    med = quant[:, 4:5]
    out = med + scale * (quant - med)
    out = np.sort(out, axis=-1)
    return (point, out)
OP_WIN = 3
OP_LEVWIN = 6
OP_FRAC = 0.6

def _operating_level(hist):
    h = np.asarray(hist, dtype=float)
    h = h[np.isfinite(h)]
    if h.size < OP_WIN or not np.all(h[-OP_WIN:] > 0):
        return None
    tail = h[-OP_LEVWIN:]
    pos = tail[tail > 0]
    if pos.size == 0:
        return None
    return float(np.median(pos))

def _operating_floor(point, quant, hist):
    lev = _operating_level(hist)
    if lev is None or lev <= 0:
        return (point, quant)
    shift = np.maximum(0.0, OP_FRAC * lev - quant[:, 4])
    if not np.any(shift > 0):
        return (point, quant)
    return (point + shift, quant + shift[:, None])

def postprocess(outputs, cases, state):
    point = np.empty((state['N'], state['H']))
    quantiles = np.empty((state['N'], state['H'], 9))
    hist = np.asarray(state['history'], dtype=float)
    obs = np.asarray(state['observed'], dtype=bool)
    for output, case in zip(outputs, cases):
        idx = case['target_indices']
        pt = np.asarray(output['point'], dtype=float)
        qt = np.asarray(output['quantiles'], dtype=float)
        for r, d in enumerate(idx):
            h = hist[d][obs[d]] if obs.shape == hist.shape else hist[d]
            p_d, q_d = _widen(pt[r], qt[r], h)
            p_d, q_d = _operating_floor(p_d, q_d, h)
            point[d] = p_d
            quantiles[d] = q_d
    nonneg = np.all(hist[np.isfinite(hist)] >= 0) if np.isfinite(hist).any() else False
    if nonneg:
        point = np.maximum(point, 0.0)
        quantiles = np.maximum(quantiles, 0.0)
    return {'point': point, 'quantiles': quantiles}

import numpy as np

def _finite(a):
    return np.asarray(a, dtype=np.float64)

def describe_targets(history, observed=None):
    Y = _finite(history)
    n, L = Y.shape
    if observed is None:
        observed = np.ones_like(Y, dtype=bool)
    observed = np.asarray(observed, dtype=bool)
    mins = np.empty(n)
    maxs = np.empty(n)
    atom = np.zeros(n)
    scale = np.ones(n)
    for i in range(n):
        y = Y[i][observed[i]] if observed[i].any() else Y[i]
        y = y[np.isfinite(y)]
        if y.size == 0:
            mins[i], maxs[i] = (-np.inf, np.inf)
            continue
        mins[i] = float(y.min())
        maxs[i] = float(y.max())
        tol = 1e-09 * max(1.0, abs(mins[i]), maxs[i] - mins[i])
        atom[i] = float((y <= mins[i] + tol).mean())
        above = y - mins[i]
        pos = above[above > tol]
        scale[i] = float(np.median(pos)) if pos.size >= 8 else float(above.mean())
        if not np.isfinite(scale[i]) or scale[i] <= 0:
            scale[i] = max(float(above.mean()), 1e-06)
    return {'min': mins, 'max': maxs, 'atom': atom, 'pos_scale': scale}

def support_floor(summary, min_atom=0.01):
    floor = np.where(summary['atom'] >= min_atom, summary['min'], -np.inf)
    return floor

def phase_profile(history, floor, period=5, window=720):
    Y = _finite(history)
    n, L = Y.shape
    w = min(window, L)
    seg = Y[:, L - w:]
    fl = np.where(np.isfinite(floor), floor, Y.min(axis=1))
    act = seg > fl[:, None] + 1e-09
    idx = np.arange(L - w, L) % period
    prof = np.zeros((n, period))
    for p in range(period):
        m = idx == p
        if m.any():
            prof[:, p] = act[:, m].mean(axis=1)
    return prof

def compress(Y, floor, scale):
    fl = np.where(np.isfinite(floor), floor, Y.min(axis=1))
    u = (Y - fl[:, None]) / scale[:, None]
    return (np.log1p(np.maximum(u, -0.999999)), fl)

def decompress(Z, floor, scale, mins):
    fl = np.where(np.isfinite(floor), floor, mins)
    return fl[..., None] + scale[..., None] * np.expm1(Z)

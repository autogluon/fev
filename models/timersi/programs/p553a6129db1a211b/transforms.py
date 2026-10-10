import numpy as np
EPS = 1e-09

def _clean(x):
    x = np.array(x, dtype=float)
    if np.all(np.isfinite(x)):
        return x
    idx = np.arange(len(x))
    good = np.isfinite(x)
    if not good.any():
        return np.zeros_like(x)
    x = np.interp(idx, idx[good], x[good])
    return x

def growth_stats(x):
    x = _clean(x)
    if len(x) < 4 or np.min(x) <= EPS:
        return (0.0, 0.0)
    g = np.diff(np.log(x))
    if len(g) < 3:
        return (0.0, 0.0)
    sd = float(np.std(g))
    snr = abs(float(np.mean(g))) / (sd + EPS)
    return (float(np.mean(g > 0)), snr)

def choose_transform(x, name=None, min_points=16, floor_ratio=0.02, min_span=1.05, min_snr=0.8, min_frac_up=0.75):
    x = _clean(x)
    if len(x) < min_points:
        return 'identity'
    med = float(np.median(np.abs(x)))
    if med <= EPS:
        return 'identity'
    if float(np.min(x)) <= floor_ratio * med:
        return 'identity'
    if float(np.max(x)) / float(np.min(x)) < min_span:
        return 'identity'
    frac_up, snr = growth_stats(x)
    if snr < min_snr or frac_up < min_frac_up:
        return 'identity'
    return 'log'

def forward(x, kind):
    x = _clean(x)
    if kind == 'log':
        return np.log(np.maximum(x, EPS))
    return x

def inverse(z, kind):
    if kind == 'log':
        return np.exp(np.clip(z, -50.0, 50.0))
    return z

def transform_matrix(mat, kinds):
    out = np.empty(np.shape(mat), dtype=float)
    for i, kind in enumerate(kinds):
        out[i] = forward(np.asarray(mat)[i], kind)
    return out

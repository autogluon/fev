import numpy as np

def plan(hist, observed=None):
    y = np.asarray(hist, float)
    m = np.isfinite(y)
    if observed is not None:
        m &= np.asarray(observed, bool)
    v = y[m]
    return {'log': bool(v.size >= 2 and np.all(v > 0))}

def forward(hist, p):
    y = np.asarray(hist, float)
    if not p['log']:
        return y
    return np.log(np.maximum(y, 1e-12))

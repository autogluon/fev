import numpy as np
EPS = 1.0

def log_forward(x, scale=EPS):
    return np.log1p(np.maximum(np.asarray(x, float), 0.0) / scale)

def log_inverse(z, scale=EPS):
    return scale * np.expm1(np.asarray(z, float))

def sort_quantiles(q):
    return np.sort(q, axis=-1)

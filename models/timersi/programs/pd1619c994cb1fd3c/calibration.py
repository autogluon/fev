import numpy as np
SPREAD = 0.9

def rescale_spread(quantiles, spread=SPREAD):
    q = np.asarray(quantiles, dtype=float)
    median = q[..., 4:5]
    out = median + spread * (q - median)
    return np.sort(out, axis=-1)

import numpy as np
BETA = 0.06
S_LO = 0.83
S_HI = 1.03

def recalibrate(quantiles, beta=BETA, s_lo=S_LO, s_hi=S_HI):
    q = np.asarray(quantiles, dtype=np.float64)
    med = q[..., 4:5]
    width = np.maximum(q[..., 8:9] - q[..., 0:1], 0.0)
    scale = np.where(np.arange(q.shape[-1]) < 4, s_lo, s_hi)
    return np.sort(med + scale * (q - med) + beta * width, axis=-1)

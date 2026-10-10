import numpy as np

def contract_multiplicative(quantiles, gamma, floor=0.5):
    q = np.asarray(quantiles, float)
    med = q[..., 4:5]
    out = med * np.power((q + floor) / (med + floor), gamma)
    return np.sort(np.maximum(out, 0.0), axis=-1)

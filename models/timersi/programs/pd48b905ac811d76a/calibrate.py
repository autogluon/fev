import numpy as np

def multiply_distribution(point, quantiles, mult):
    return (point * mult, quantiles * mult[:, None])

def rescale_width(point, quantiles, factor):
    med = quantiles[:, 4:5]
    return med + factor[:, None] * (quantiles - med)

def horizon_width_factor(quantiles, s0, beta, lo=0.35, hi=1.25):
    sig = (quantiles[:, 8] - quantiles[:, 0]) / 2.5631
    sig = np.maximum(sig, 1e-09)
    ratio = sig[0:1] / sig
    return np.clip(s0 * ratio ** beta, lo, hi)

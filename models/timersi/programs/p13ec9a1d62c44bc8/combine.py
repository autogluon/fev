import numpy as np

def lead_ramp(horizon, w_near, w_far, ramp_hours=120.0):
    t = np.clip(np.arange(horizon, dtype=float) / float(ramp_hours), 0.0, 1.0)
    return w_near + (w_far - w_near) * t

def blend_distribution(point, quantiles, specialist, weight, width=1.0):
    point = np.asarray(point, float)
    quantiles = np.asarray(quantiles, float)
    spec = np.asarray(specialist, float)
    spec = np.where(np.isfinite(spec), spec, point)
    w = np.asarray(weight, float)
    if w.ndim == 1:
        w = w[None, :]
    median = quantiles[..., 4]
    new_point = (1.0 - w) * point + w * spec
    new_median = (1.0 - w) * median + w * spec
    out = new_median[..., None] + width * (quantiles - median[..., None])
    return (new_point, np.sort(out, axis=-1))

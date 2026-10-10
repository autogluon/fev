import numpy as np

def sanitize(q, fallback):
    q = np.array(q, dtype=float, copy=True)
    bad = ~np.isfinite(q)
    if bad.any():
        q[bad] = np.asarray(fallback, dtype=float)[bad]
    return q

def clamp_to_plausible(q_native, fit, horizon, n_sigma=8.0, min_band=0.5):
    h = np.arange(1, horizon + 1, dtype=float)
    eff = max(fit.get('eff', max(fit['n'], 1.0)), 1.0)
    sd = max(fit['sigma'], 1e-09) * np.sqrt(h + h ** 2 / eff)
    band = np.maximum(n_sigma * sd, min_band * np.abs(fit['drift']) * h + 1e-09)
    centre = fit['level'] + fit['drift'] * h
    lo = (centre - band)[:, None]
    hi = (centre + band)[:, None]
    return np.clip(q_native, lo, hi)

def blend_quantiles(q_native, q_anchor, weight):
    w = float(np.clip(weight, 0.0, 1.0))
    out = (1.0 - w) * np.asarray(q_native, float) + w * np.asarray(q_anchor, float)
    return np.sort(out, axis=-1)

def widen_floor(q, q_anchor, floor_ratio=0.0):
    if floor_ratio <= 0.0:
        return q
    med = q[..., 4:5]
    dev = q - med
    anchor_dev = q_anchor - q_anchor[..., 4:5]
    scaled = np.where(np.abs(anchor_dev) * floor_ratio > np.abs(dev), anchor_dev * floor_ratio, dev)
    return np.sort(med + scaled, axis=-1)
NORM_HW = 2.0 * 1.2815515655446004

def moment_mix(q_native, q_anchor, w_centre, w_spread=None, disagreement=1.0):
    ws = w_centre if w_spread is None else w_spread
    qn = np.asarray(q_native, float)
    qa = np.asarray(q_anchor, float)
    mn = qn[..., 4:5]
    ma = qa[..., 4:5]
    sn = np.maximum((qn[..., 8:9] - qn[..., 0:1]) / NORM_HW, 1e-12)
    sa = np.maximum((qa[..., 8:9] - qa[..., 0:1]) / NORM_HW, 1e-12)
    mu = (1.0 - w_centre) * mn + w_centre * ma
    var = (1.0 - ws) * sn ** 2 + ws * sa ** 2 + disagreement * w_centre * (1.0 - w_centre) * (mn - ma) ** 2
    sd = np.sqrt(np.maximum(var, 1e-24))
    shape = (1.0 - ws) * (qn - mn) / sn + ws * (qa - ma) / sa
    hw = np.maximum((shape[..., 8:9] - shape[..., 0:1]) / NORM_HW, 1e-12)
    shape = shape / hw
    return np.sort(mu + sd * shape, axis=-1)

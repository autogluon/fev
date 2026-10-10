import numpy as np
QLEV = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

def clean_history(hist, observed):
    y = np.asarray(hist, dtype=float).copy()
    ok = np.asarray(observed, dtype=bool) & np.isfinite(y)
    if not ok.any():
        return (np.zeros_like(y), ok)
    idx = np.arange(y.size)
    first = idx[ok][0]
    last_val = y[first]
    for t in range(y.size):
        if t < first or not ok[t]:
            y[t] = last_val
        else:
            last_val = y[t]
    return (y, ok)

def cumulative_profile(y):
    y = np.asarray(y, float)
    L = y.size
    out = {'L': int(L), 'last': float(y[-1]) if L else 0.0, 'min': float(y.min()) if L else 0.0, 'max': float(y.max()) if L else 0.0}
    if L < 2:
        out.update(mono_frac=1.0, is_cumulative=True, naive_mae=0.0, int_frac=1.0, drop_mag=0.0)
        return out
    d = np.diff(y)
    out['naive_mae'] = float(np.mean(np.abs(d)))
    out['mono_frac'] = float(np.mean(d >= -1e-09))
    rng = max(y.max() - y.min(), 1e-09)
    out['drop_mag'] = float(-min(d.min(), 0.0) / rng)
    out['int_frac'] = float(np.mean(np.abs(y - np.round(y)) < 1e-06))
    out['is_cumulative'] = bool(out['mono_frac'] >= 0.98 and y.min() >= -1e-09 and (out['int_frac'] >= 0.9))
    return out

def enforce_quantile_sort(q):
    return np.sort(q, axis=-1)

def project_cumulative(point, quant, floor):
    point = np.maximum(np.asarray(point, float), floor)
    quant = np.maximum(np.asarray(quant, float), floor)
    point = np.maximum.accumulate(point, axis=-1)
    quant = np.maximum.accumulate(quant, axis=-2)
    quant = enforce_quantile_sort(quant)
    point = np.clip(point, quant[..., 0], quant[..., -1])
    return (point, quant)

def integrate(new_point, new_quant, y_last, width_beta=0.85, lam=1.0):
    H = new_point.shape[-1]
    med = new_quant[..., 4]
    cum_med = y_last + np.cumsum(med, axis=-1)
    cum_pt = y_last + np.cumsum(new_point, axis=-1)
    dev = new_quant - med[..., None]
    cdev = np.cumsum(dev, axis=-2)
    h = np.arange(1, H + 1, dtype=float)
    shrink = (h ** (width_beta - 1.0))[:, None]
    cum_q = cum_med[..., None] + lam * cdev * shrink
    return (cum_pt, cum_q)

def softplus_floor(x, floor, scale):
    z = (x - floor) / max(scale, 1e-09)
    return floor + scale * np.logaddexp(0.0, z)
_ZQ = np.array([-1.2815515655446004, -0.8416212335729143, -0.5244005127080409, -0.2533471031357997, 0.0, 0.2533471031357997, 0.5244005127080409, 0.8416212335729143, 1.2815515655446004])

def onset_weight(profile, inc_past, max_level=10.0, max_len=14.0):
    if not profile.get('is_cumulative'):
        return 0.0
    yL = profile['last']
    if yL < 1.0:
        return 0.0
    nz = int(np.sum(np.asarray(inc_past) > 0))
    w_level = np.clip((max_level - yL) / (max_level - 2.0), 0.0, 1.0)
    w_len = np.clip((max_len - profile['L']) / 6.0, 0.0, 1.0)
    w_flat = np.clip((3.0 - nz) / 2.0, 0.0, 1.0)
    return float(w_level * w_len * w_flat)

def onset_fan(y_last, H, med_mult=600.0, sig=1.2, k_shape=3.2):
    anchor = max(float(y_last), 1.0)
    h = np.arange(1, H + 1, dtype=float)
    tau = k_shape * (1.0 - np.exp(-h / k_shape))
    shape = tau / tau[-1]
    return anchor * np.exp(np.outer(shape, np.log(med_mult) + _ZQ * sig))

def log_blend(a, b, w):
    a = np.maximum(a, 0.0)
    b = np.maximum(b, 0.0)
    return np.expm1((1.0 - w) * np.log1p(a) + w * np.log1p(b))

def log_rate_innovation_sd(inc_past, window=26, min_obs=8):
    d = np.asarray(inc_past, dtype=float)
    d = d[np.isfinite(d)]
    if d.size < min_obs + 1:
        return None
    d = d[-window:]
    r = np.log1p(np.maximum(d, 0.0))
    e = np.diff(r, n=2) / np.sqrt(6.0)
    e = e[np.isfinite(e)]
    if e.size < min_obs:
        return None
    mad = np.median(np.abs(e - np.median(e)))
    return float(1.4826 * mad)

def widen_increment_band(inc_q, v, zq=_ZQ, v_max=0.45, damp=0.8):
    if v is None or not np.isfinite(v) or v <= 0:
        return inc_q
    H = inc_q.shape[0]
    h = np.arange(1, H + 1, dtype=float)
    s = np.minimum(damp * v * np.sqrt(h), v_max)[:, None]
    m = np.maximum(inc_q[:, 4:5], 1e-09)
    floor_lo = m * np.exp(zq[None, :] * s)
    out = np.where(zq[None, :] >= 0, np.maximum(inc_q, floor_lo), np.minimum(inc_q, floor_lo))
    return np.maximum(out, 0.0)

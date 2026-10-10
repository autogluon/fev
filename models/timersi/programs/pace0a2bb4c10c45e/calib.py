import numpy as np

def widen(quant, factor, floor=None):
    med = quant[..., 4:5]
    out = med + np.asarray(factor)[..., None] * (quant - med)
    if floor is not None:
        out = np.maximum(out, floor)
    return np.sort(out, axis=-1)
DISCOUNT_NAMES = ['type_%d_discount' % i for i in range(7)]

def promo_depth(matrix, names):
    rows = [matrix[i] for i, nm in enumerate(names) if nm in DISCOUNT_NAMES]
    if not rows:
        return None
    return np.nanmax(np.stack([np.nan_to_num(r, nan=0.0) for r in rows]), axis=0)

def young_spread(n_observed, tail_observed, w_young=1.0, w_tail=0.5, cap=120.0):
    young = max(0.0, (cap - float(n_observed)) / cap)
    sparse = max(0.0, 1.0 - float(tail_observed))
    return 1.0 + w_young * young + w_tail * sparse

def promo_correction(point, quant, sales, observed, past_depth, future_depth, weight=0.5, max_events=40.0, thr=0.03, window=730, clip=(0.5, 3.0)):
    if future_depth is None or past_depth is None:
        return (point, quant)
    obs = np.asarray(observed, dtype=bool)
    events = int(((past_depth > thr) & obs).sum())
    if float(np.nanmax(future_depth)) <= thr or events >= max_events:
        return (point, quant)
    w = weight * max(0.0, 1.0 - events / float(max_events))
    if w <= 0:
        return (point, quant)
    z = np.log1p(np.maximum(np.asarray(sales, dtype=float), 0.0))
    recent = np.zeros(len(z), dtype=bool)
    recent[-window:] = True
    base_sel = obs & recent & (past_depth <= thr)
    if base_sel.sum() < 20:
        base_sel = obs & (past_depth <= thr)
    if base_sel.sum() < 10:
        return (point, quant)
    base_hist = float(np.median(z[base_sel]))
    zp = np.log1p(np.maximum(point, 0.0))
    quiet = future_depth <= thr
    anchor = float(np.median(zp[quiet])) if quiet.sum() >= 2 else base_hist
    out_p, out_q = (point.copy(), quant.copy())
    for h in range(len(point)):
        d = float(future_depth[h])
        if d <= thr:
            continue
        sel = obs & (past_depth > thr) & (np.abs(past_depth - d) <= 0.06)
        if sel.sum() < 3:
            sel = obs & (past_depth > thr)
        if sel.sum() < 3:
            continue
        uplift = float(np.median(z[sel])) - base_hist
        if not np.isfinite(uplift) or uplift <= 0:
            continue
        blended = (1.0 - w) * zp[h] + w * (anchor + uplift)
        ratio = float(np.clip((np.expm1(blended) + 1e-06) / (np.expm1(zp[h]) + 1e-06), clip[0], clip[1]))
        out_p[h] = point[h] * ratio
        out_q[h] = quant[h] * ratio
    return (out_p, np.sort(out_q, axis=-1))

def young_trend_factor(sales, observed, horizon, cap=120.0, damp=0.6, max_slope=0.03, window=56, min_points=10):
    obs = np.asarray(observed, dtype=bool)
    n_obs = int(obs.sum())
    if n_obs >= cap or n_obs < min_points:
        return None
    idx = np.where(obs)[0][-window:]
    z = np.log1p(np.maximum(np.asarray(sales, dtype=float)[idx], 0.0))
    t = idx.astype(float) - idx.mean()
    den = float((t * t).sum())
    if den <= 0 or not np.all(np.isfinite(z)):
        return None
    slope = float((t * (z - z.mean())).sum() / den)
    slope = float(np.clip(slope, -max_slope, max_slope))
    if abs(slope) < 1e-06:
        return None
    return np.exp(damp * slope * np.arange(1, horizon + 1))

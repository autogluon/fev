import numpy as np
LEVEL_GAIN = 0.35
SLOPE_GAIN = 0.25
LEVEL_CAP = 0.06
Z_LO, Z_HI = (0.7, 1.25)
SPREAD_BLEND = 0.5
NORM = np.array([-1.2815515655446004, -0.8416212335729143, -0.5244005127080409, -0.2533471031357997, 0.0, 0.2533471031357997, 0.5244005127080409, 0.8416212335729143, 1.2815515655446004])

def residuals(aux):
    out = []
    for a in aux:
        p = np.asarray(a['point'], dtype=float)
        t = np.asarray(a['truth'], dtype=float)
        ok = np.isfinite(p) & np.isfinite(t) & (p > 0) & (t > 0)
        if ok.sum() < 2:
            continue
        r = np.full(p.shape, np.nan)
        r[ok] = np.log(t[ok] / p[ok])
        out.append({'r': r, 'w': float(a.get('weight', 1.0)), 'role': a.get('role', '?'), 'q': np.asarray(a['quantiles'], dtype=float), 'truth': t, 'point': p})
    return out

def _wls_line(h, y, w):
    m = np.isfinite(y) & (w > 0)
    if m.sum() < 3:
        return (float(np.nansum(y[m] * w[m]) / max(1e-09, np.nansum(w[m]))) if m.any() else 0.0, 0.0)
    x = h[m] - h[m].mean()
    ww = w[m]
    sw = ww.sum()
    a = float((ww * y[m]).sum() / sw)
    sxx = float((ww * x * x).sum())
    c = float((ww * x * y[m]).sum() / sxx) if sxx > 1e-09 else 0.0
    return (a, c)

def level_correction(res, H, level_gain=LEVEL_GAIN, slope_gain=SLOPE_GAIN, cap=LEVEL_CAP):
    info = {'n_origins': len(res)}
    if not res:
        return (np.zeros(H), info)
    h = np.arange(H, dtype=float)
    per = []
    for d in res:
        r = d['r'][:H] if d['r'].size >= H else np.pad(d['r'], (0, H - d['r'].size), constant_values=np.nan)
        per.append((r, d['w']))
    R = np.array([p[0] for p in per])
    W = np.array([p[1] for p in per])[:, None] * np.isfinite(R)
    with np.errstate(invalid='ignore'):
        flat_y = np.nansum(np.where(np.isfinite(R), R, 0.0) * W, axis=0) / np.maximum(1e-09, W.sum(axis=0))
    flat_w = W.sum(axis=0)
    a, c = _wls_line(h, flat_y, flat_w)
    means = np.array([np.nanmean(r) for r, _ in per if np.isfinite(r).any()])
    if means.size >= 2:
        sd = float(np.std(means))
        rel = float(abs(np.mean(means)) / (abs(np.mean(means)) + sd + 1e-09))
        rel = 0.35 + 0.65 * rel
    else:
        rel = 0.5
    info.update({'a': float(a), 'c': float(c), 'rel': rel, 'origin_means': [float(m) for m in means]})
    b = rel * (level_gain * a + slope_gain * c * (h - h.mean()))
    b = np.clip(b, -cap, cap)
    info['b'] = [float(v) for v in b]
    return (b, info)

def spread_scale(res, blend=SPREAD_BLEND, lo=Z_LO, hi=Z_HI):
    zs, ws = ([], [])
    for d in res:
        q = d['q']
        if q.ndim != 2 or q.shape[-1] != 9:
            continue
        med = q[:, 4]
        halfw = 0.5 * (q[:, 7] - q[:, 1])
        sig = halfw / NORM[7]
        ok = np.isfinite(sig) & (sig > 1e-06) & np.isfinite(d['truth'])
        if not ok.any():
            continue
        z = (d['truth'][ok] - med[ok]) / sig[ok]
        zs.append(np.abs(z))
        ws.append(np.full(z.shape, d['w']))
    if not zs:
        return (1.0, {'measured': None})
    az = np.concatenate(zs)
    aw = np.concatenate(ws)
    if az.size < 6:
        return (1.0, {'measured': None, 'n': int(az.size)})
    measured = float(np.sum(aw * az) / np.sum(aw) / np.sqrt(2.0 / np.pi))
    s = 1.0 + blend * (float(np.clip(measured, lo, hi)) - 1.0)
    return (s, {'measured': measured, 'n': int(az.size), 'scale': s})

def apply(point, quantiles, b, spread=1.0):
    p = np.asarray(point, dtype=float)
    q = np.asarray(quantiles, dtype=float)
    f = np.exp(np.asarray(b, dtype=float))[:, None]
    q = q * f
    p = p * f[:, 0]
    if abs(spread - 1.0) > 1e-09:
        med = q[:, 4:5]
        q = med + (q - med) * spread
    return (p, np.sort(q, axis=-1))

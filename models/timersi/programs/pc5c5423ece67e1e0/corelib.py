import numpy as np
QUANTILES = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
_NORMAL_Z = np.array([-1.2815515655446004, -0.8416212335729143, -0.5244005127080407, -0.2533471031357997, 0.0, 0.2533471031357997, 0.5244005127080407, 0.8416212335729143, 1.2815515655446004])

def ewma_level(y, alpha):
    if len(y) == 0:
        return 0.0
    if alpha <= 1e-09:
        return float(np.mean(y))
    w = (1.0 - alpha) ** np.arange(len(y) - 1, -1, -1)
    s = w.sum()
    if not np.isfinite(s) or s <= 0:
        return float(y[-1])
    return float(np.dot(w, y) / s)

def centred_slope(y):
    n = len(y)
    if n < 6:
        return 0.0
    t = np.arange(n) - (n - 1) / 2.0
    den = float(np.dot(t, t))
    if den <= 0:
        return 0.0
    return float(np.dot(t, y - float(np.mean(y))) / den)

def eb_pool(levels, within_var, n_obs):
    k = len(levels)
    if k < 3 or n_obs < 2 or (not np.isfinite(within_var)) or (within_var <= 0):
        return (1.0, float(np.mean(levels)) if k else 0.0)
    between = float(np.var(levels, ddof=1))
    noise = within_var / float(n_obs)
    tau2 = max(between - noise, 0.0)
    w = tau2 / (tau2 + noise + 1e-12)
    w = w * (1.0 - 1.0 / max(k - 1.0, 1.0)) if k < 6 else w
    return (float(min(max(w, 0.0), 1.0)), float(np.mean(levels)))

def pooled_rho(contexts, shrink=20.0):
    num = 0.0
    den = 0.0
    n = 0
    for z in contexts:
        if len(z) < 4:
            continue
        d = z - float(np.mean(z))
        num += float(np.dot(d[1:], d[:-1]))
        den += float(np.dot(d, d))
        n += len(d) - 1
    if den <= 0 or n < 8:
        return 0.0
    r = num / den
    r = r * n / (n + shrink)
    return float(np.clip(r, -0.6, 0.9))

def centre_path(contexts, i, alpha, pool, damp, horizon, rho=0.0):
    y = contexts[i]
    m = ewma_level(y, alpha)
    if pool and len(contexts) >= 3:
        levels = np.array([ewma_level(z, alpha) for z in contexts], dtype=float)
        vs = [float(np.var(z, ddof=1)) for z in contexts if len(z) > 2]
        within = float(np.mean(vs)) if vs else 0.0
        w, gm = eb_pool(levels, within, len(y))
        m = gm + w * (m - gm)
    hv = np.arange(1, horizon + 1)
    path = np.full(horizon, float(m))
    if damp > 0.0:
        path = path + centred_slope(y) * np.cumsum(damp ** hv)
    if rho != 0.0 and len(y) >= 4:
        path = path + rho ** hv * (float(y[-1]) - float(np.mean(y)))
    return path

def rolling_backtest(histories, candidates, horizon, min_ctx=4, min_ctx_frac=3):
    k = len(histories)
    scales = []
    for h in histories:
        d = np.abs(np.diff(h)) if len(h) > 1 else np.array([1.0])
        s = float(np.mean(d)) if len(d) else 1.0
        scales.append(s if np.isfinite(s) and s > 1e-09 else max(float(np.mean(np.abs(h))), 1.0))
    acc = {c: [] for c in candidates}
    for i, h in enumerate(histories):
        L = len(h)
        lo = max(min_ctx, L // max(min_ctx_frac, 1))
        if L - lo < 1:
            continue
        for o in range(lo, L):
            idx = [j for j in range(k) if len(histories[j]) >= o]
            ctxs = [histories[j][:o] for j in idx]
            try:
                ii = idx.index(i)
            except ValueError:
                continue
            hh = min(horizon, L - o)
            hv = np.arange(1, hh + 1, dtype=float)
            rho_o = pooled_rho(ctxs)
            for c in candidates:
                f = centre_path(ctxs, ii, c[0], c[1], c[2], hh, rho_o if c[3] else 0.0)
                acc[c].append(np.stack([hv, h[o:o + hh] - f, np.full(hh, scales[i]), np.full(hh, float(o))], axis=1))
    out = {}
    for c, v in acc.items():
        out[c] = np.concatenate(v, axis=0) if v else np.zeros((0, 4))
    return out

def cornish_fisher(skew, kurt):
    s = float(np.clip(skew, -1.0, 1.0))
    k = float(np.clip(kurt, -1.0, 3.0))
    z = _NORMAL_Z
    cf = z + (z ** 2 - 1) * s / 6.0 + (z ** 3 - 3 * z) * k / 24.0 - (2 * z ** 3 - 5 * z) * s ** 2 / 36.0
    return np.sort(cf)

def variance_growth(records, horizon, prior=200.0, origin_weight=True):
    hv = np.arange(1, horizon + 1, dtype=float)
    if len(records) < 8:
        return np.ones(horizon)
    hb = records[:, 0]
    e2 = records[:, 1] ** 2
    ow = np.sqrt(records[:, 3] / max(float(np.max(records[:, 3])), 1.0)) if origin_weight else np.ones(len(records))
    X = np.stack([np.ones_like(hb), hb - 1.0], axis=1) * ow[:, None]
    try:
        coef, _, _, _ = np.linalg.lstsq(X, e2 * ow, rcond=None)
    except Exception:
        return np.ones(horizon)
    a0, b0 = (float(coef[0]), float(coef[1]))
    if not np.isfinite(a0) or a0 <= 0 or (not np.isfinite(b0)):
        return np.ones(horizon)
    b0 = max(b0, 0.0)
    n = len(hb)
    ratio = n / (n + prior) * b0 / a0
    g = np.sqrt(np.maximum(1.0 + ratio * (hv - 1.0), 1e-09))
    return np.minimum(g, np.sqrt(hv))

def default_candidates():
    return [(a, p, d, r) for a in (0.0, 0.05, 0.1, 0.2, 0.35, 0.6, 1.0) for p in (True, False) for d in (0.0, 0.6) for r in (False, True)]

def eb_scale(contexts, anchors):
    k = len(contexts)
    ns = np.array([len(z) for z in contexts], dtype=float)
    v = np.array([max(float(np.mean((contexts[i] - anchors[i]) ** 2)), 1e-12) for i in range(k)])
    lv = np.log(v)
    if k < 4 or ns.min() < 5:
        return (np.ones(k), 0.0)
    noise = float(np.mean(2.0 / np.maximum(ns - 1.0, 1.0)))
    between = float(np.var(lv, ddof=1))
    tau2 = max(between - noise, 0.0)
    w = tau2 / (tau2 + noise + 1e-12)
    w = w * (1.0 - 1.0 / max(k - 1.0, 1.0)) if k < 6 else w
    factor = np.exp(0.5 * w * (lv - float(np.mean(lv))))
    factor = factor / float(np.sqrt(np.mean(factor ** 2)))
    return (factor, float(w))

def fit_pool(histories, horizon, temperature=0.02, emp_prior=60.0, candidates=None, origin_weight=True, min_ctx_frac=3):
    k = len(histories)
    H = int(horizon)
    if candidates is None:
        candidates = default_candidates()
    usable = all((len(h) >= 5 for h in histories))
    if not usable:
        candidates = [(0.0, k >= 3, 0.0, False), (1.0, False, 0.0, False)]
    rho = pooled_rho(histories)
    records = rolling_backtest(histories, candidates, H, min_ctx_frac=min_ctx_frac)

    def _loss(rec):
        if not len(rec):
            return float('inf')
        w = rec[:, 3] / max(float(np.max(rec[:, 3])), 1.0) if origin_weight else np.ones(len(rec))
        return float(np.sum(w * np.abs(rec[:, 1]) / rec[:, 2]) / max(np.sum(w), 1e-09))
    losses = np.array([_loss(records[c]) for c in candidates], dtype=float)
    if not np.any(np.isfinite(losses)):
        weights = np.zeros(len(candidates))
        weights[0] = 1.0
    else:
        losses = np.where(np.isfinite(losses), losses, np.nanmax(losses[np.isfinite(losses)]) * 10)
        best = float(losses.min())
        weights = np.exp(-(losses - best) / max(temperature * max(best, 1e-09), 1e-12))
        weights = weights / weights.sum()
    keep = weights > 0.0001
    centres = np.zeros((k, H))
    anchors = np.zeros(k)
    for ci, (c, w) in enumerate(zip(candidates, weights)):
        if not keep[ci]:
            continue
        r = rho if c[3] else 0.0
        for i in range(k):
            centres[i] += w * centre_path(histories, i, c[0], c[1], c[2], H, r)
            anchors[i] += w * centre_path(histories, i, c[0], c[1], c[2], 1, 0.0)[0]
    alpha_eff = float(np.sum(weights * np.array([c[0] for c in candidates])))
    eff_par = float(np.sum(weights * np.array([(1.0 / k if c[1] else 1.0) + (1.0 if c[2] > 0 else 0.0) for c in candidates])))
    res = np.concatenate([histories[i] - anchors[i] for i in range(k)])
    n_mean = float(np.mean([len(h) for h in histories]))
    res = res * np.sqrt(n_mean / max(n_mean - eff_par, 1.0))
    sigma = float(np.std(res))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = max(float(np.std(np.concatenate(histories))), 1e-06)
    if k >= 3:
        vs = [float(np.var(z, ddof=1)) for z in histories if len(z) > 2]
        within = float(np.mean(vs)) if vs else 0.0
        w_lvl, gm = eb_pool(anchors, within, n_obs=float(np.mean([len(h) for h in histories])))
        shift = gm + w_lvl * (anchors - gm) - anchors
        centres = centres + shift[:, None]
        anchors = anchors + shift
    else:
        w_lvl = 1.0
    pooled_rec = np.concatenate([records[c] for ci, c in enumerate(candidates) if keep[ci]], axis=0) if keep.any() else np.zeros((0, 3))
    growth = variance_growth(pooled_rec, H, origin_weight=origin_weight)
    growth = np.maximum(growth, np.sqrt(1.0 + (np.arange(1, H + 1) - 1) * alpha_eff ** 2)) if alpha_eff > 0.5 else growth
    sd = np.sqrt(sigma ** 2 * growth ** 2 + sigma ** 2 * eff_par / max(n_mean, 1.0))
    z = res / sigma
    n = len(z)
    w_emp = n / (n + emp_prior)
    skew = float(np.mean(z ** 3)) if n > 8 else 0.0
    kurt = float(np.mean(z ** 4) - 3.0) if n > 8 else 0.0
    shape = np.sort(w_emp * np.quantile(z, QUANTILES) + (1.0 - w_emp) * cornish_fisher(skew, kurt))
    scale_factor, w_scale = eb_scale(histories, anchors)
    info = {'alpha_eff': alpha_eff, 'eff_par': eff_par, 'sigma': sigma, 'rho': rho, 'w_level': float(w_lvl), 'scale_factor': [float(v) for v in scale_factor], 'w_scale': w_scale, 'growth': [float(v) for v in growth], 'skew': skew, 'kurt': kurt, 'n_resid': int(n), 'w_emp': float(w_emp), 'shape': [float(v) for v in shape], 'best_candidates': [[list(candidates[j]), float(losses[j]), float(weights[j])] for j in np.argsort(losses)[:5]]}
    sd_item = scale_factor[:, None] * sd[None, :]
    return (centres, sd_item, shape, info)

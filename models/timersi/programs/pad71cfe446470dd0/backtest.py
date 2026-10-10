import numpy as np
CANDIDATE_OFFSETS = (168, 336)
FALLBACK_OFFSETS = (504, 672)
MIN_OBS_FRACTION = 0.4
MIN_TARGET_OBS = 48
MAX_ORIGINS = 2
RECENCY = {72: 0.8, 168: 0.65, 336: 0.35, 504: 0.22, 672: 0.15}
PARTIAL_OFFSET = 72
MIN_PARTIAL_POINTS = 24
HORIZON_TAU_FRACTION = 0.5

def _usable(obs, L, H, off):
    if off < H or L - off < 6 * 168:
        return None
    win = obs[:, L - off:L - off + H]
    frac = float(win.mean())
    if frac < MIN_OBS_FRACTION:
        return None
    if int(win.sum(axis=1).max()) < MIN_TARGET_OBS:
        return None
    return frac

def pick_origins(observed, L, H, max_origins=MAX_ORIGINS):
    obs = np.asarray(observed, bool)
    chosen = []
    for off in CANDIDATE_OFFSETS:
        frac = _usable(obs, L, H, off)
        if frac is not None:
            chosen.append((off, frac))
        if len(chosen) >= max_origins:
            return chosen
    if not chosen:
        for off in FALLBACK_OFFSETS:
            frac = _usable(obs, L, H, off)
            if frac is not None:
                chosen.append((off, frac))
            if len(chosen) >= max_origins:
                break
    return chosen

def _weighted_median(x, w):
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    cw = cw / cw[-1]
    return float(np.interp(0.5, cw, x[o]))

def log_bias(truth, mask, point, floor=0.001, min_points=MIN_TARGET_OBS, tau_fraction=HORIZON_TAU_FRACTION):
    D = truth.shape[0]
    H = truth.shape[1]
    tau = max(1.0, tau_fraction * H)
    rec = np.exp(-(H - 1 - np.arange(H)) / tau)
    out = np.full(D, np.nan)
    cnt = np.zeros(D)
    for d in range(D):
        m = mask[d] & np.isfinite(truth[d]) & np.isfinite(point[d]) & (point[d] > 0)
        cnt[d] = m.sum()
        if m.sum() >= min_points:
            r = np.log(np.clip(truth[d][m], floor, None)) - np.log(np.clip(point[d][m], floor, None))
            out[d] = _weighted_median(r, rec[m])
    return (out, cnt)

def combine(records, D, pool_weight=1.0):
    num = np.zeros(D)
    den = np.zeros(D)
    for off, frac, b, c in records:
        w0 = RECENCY.get(off, 0.2)
        for d in range(D):
            if np.isfinite(b[d]):
                w = w0 * min(1.0, c[d] / 84.0)
                num[d] += w * b[d]
                den[d] += w
    per = np.where(den > 0, num / np.maximum(den, 1e-09), np.nan)
    if not np.isfinite(per).any():
        return (np.zeros(D), False)
    pooled = float(np.nanmean(per))
    out = np.where(np.isfinite(per), pool_weight * pooled + (1 - pool_weight) * per, pooled)
    return (out, True)

def pick_partial(observed, L, H, offset=PARTIAL_OFFSET):
    obs = np.asarray(observed, bool)
    if offset >= H or offset >= L or L - offset < 6 * 168:
        return None
    win = obs[:, L - offset:L]
    if int(win.sum(axis=1).max()) < MIN_PARTIAL_POINTS:
        return None
    return (offset, float(win.mean()))

def partial_log_bias(truth, mask, point, floor=0.001, min_points=MIN_PARTIAL_POINTS):
    D = truth.shape[0]
    K = truth.shape[1]
    out = np.full(D, np.nan)
    cnt = np.zeros(D)
    for d in range(D):
        m = mask[d] & np.isfinite(truth[d]) & np.isfinite(point[d][:K]) & (point[d][:K] > 0)
        cnt[d] = m.sum()
        if m.sum() >= min_points:
            r = np.log(np.clip(truth[d][m], floor, None)) - np.log(np.clip(point[d][:K][m], floor, None))
            out[d] = float(np.median(r))
    return (out, cnt)
QUANTILE_LEVELS = np.arange(1, 10) / 10.0

def upper_tail_coverage(truth, mask, quantiles, min_points=48, floor=0.001):
    q = np.sort(np.asarray(quantiles, float), axis=2)
    D = truth.shape[0]
    K = min(truth.shape[1], q.shape[1])
    cov = []
    for d in range(D):
        m = mask[d][:K] & np.isfinite(truth[d][:K]) & (q[d][:K, 4] > 0)
        if m.sum() < min_points:
            continue
        tr = truth[d][:K][m]
        med = np.median(np.log(np.clip(tr, floor, None)) - np.log(np.clip(q[d][:K, 4][m], floor, None)))
        cov.append(float(np.mean(tr <= q[d][:K, 8][m] * np.exp(med))))
    return float(np.mean(cov)) if cov else np.nan

def pinball_by_target(truth, mask, quantiles, tau_fraction=HORIZON_TAU_FRACTION):
    D, H = (truth.shape[0], truth.shape[1])
    q = np.sort(np.asarray(quantiles, float), axis=2)
    tau = max(1.0, tau_fraction * H)
    rec = np.exp(-(H - 1 - np.arange(H)) / tau)
    out = np.full(D, np.nan)
    for d in range(D):
        m = mask[d] & np.isfinite(truth[d])
        if m.sum() < MIN_TARGET_OBS:
            continue
        y = truth[d][m][:, None]
        qq = q[d][m, :]
        w = rec[m][:, None]
        diff = y - qq
        pl = np.maximum(QUANTILE_LEVELS[None, :] * diff, (QUANTILE_LEVELS[None, :] - 1) * diff)
        out[d] = float((pl * w).sum() / (w.sum() * len(QUANTILE_LEVELS)))
    return out

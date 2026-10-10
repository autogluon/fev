import numpy as np
QL = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

def robust_sigma(h):
    if len(h) < 3:
        return 0.0
    d = np.diff(np.asarray(h, dtype=float))
    s = float(np.median(np.abs(d - np.median(d))) * 1.4826)
    if not np.isfinite(s) or s <= 0:
        s = float(np.std(d))
    return s if np.isfinite(s) and s > 0 else 0.0

def anchor_level(h, k=8):
    h = np.asarray(h, dtype=float)
    k = int(min(max(k, 1), len(h)))
    return float(np.median(h[-k:]))

def launch_index(h):
    nz = np.nonzero(np.asarray(h, dtype=float) > 0)[0]
    return int(nz[0]) if len(nz) else -1

def activity_probs(post_bin, H, pseudo=0.5):
    b = np.asarray(post_bin, dtype=float)
    p = float(b.mean())
    a, c = (b[:-1], b[1:])
    n11 = float(np.sum((a == 1) & (c == 1)))
    n10 = float(np.sum((a == 1) & (c == 0)))
    n01 = float(np.sum((a == 0) & (c == 1)))
    n00 = float(np.sum((a == 0) & (c == 0)))
    p11 = (n11 + pseudo * p) / (n11 + n10 + pseudo)
    p01 = (n01 + pseudo * p) / (n01 + n00 + pseudo)
    cur = float(b[-1])
    out = np.empty(H)
    for j in range(H):
        cur = cur * p11 + (1.0 - cur) * p01
        out[j] = cur
    return np.clip(out, 0.0, 1.0)

def mixture_quantiles(p, active_q):
    out = np.zeros(9)
    for j, q in enumerate(QL):
        if q <= 1.0 - p:
            out[j] = 0.0
        else:
            u = (q - (1.0 - p)) / max(p, 1e-09)
            out[j] = float(np.interp(u, QL, active_q))
    return out

def paused_forecast(h, H, pseudo=0.5, lookback=26):
    h = np.asarray(h, dtype=float)
    li = launch_index(h)
    if li < 0:
        return None
    post = h[li:]
    pb = (post > 0).astype(float)
    if pb[-1] == 1 or len(pb) < 3:
        return None
    pa = activity_probs(pb, H, pseudo)
    act = post[post > 0][-lookback:]
    if len(act) == 0:
        return None
    aq = np.quantile(act, QL)
    Qn = np.stack([mixture_quantiles(pa[j], aq) for j in range(H)])
    return (Qn, Qn[:, 4].copy())

def level_trend_correction(point, h, H, lam_c0=30.0, lam_cap=0.7, kappa_trend=0.5, kanch=8):
    h = np.asarray(h, dtype=float)
    C = len(h)
    hh = np.arange(1, H + 1, dtype=float)
    hc = hh - hh.mean()
    a = anchor_level(h, kanch)
    sig = robust_sigma(h)
    p = np.asarray(point, dtype=float)
    m = float(p.mean())
    b1 = float(np.dot(hc, p - m) / np.dot(hc, hc))
    resid = p - m - b1 * hc
    lam = min(lam_cap, C / (C + lam_c0))
    m_new = a + lam * (m - a)
    b1_new = b1
    if sig > 0 and kappa_trend is not None:
        lim = kappa_trend * sig * np.sqrt(H) / max(H - 1, 1)
        b1_new = float(np.clip(b1, -lim, lim))
    return m_new + b1_new * hc + resid

def activity_plan(h, H, min_active=4, pseudo=0.5):
    h = np.asarray(h, dtype=float)
    li = launch_index(h)
    if li < 0:
        return None
    post = h[li:]
    idx = np.nonzero(h > 0)[0]
    if len(idx) < min_active:
        return None
    pb = (post > 0).astype(float)
    if pb.min() == 1 and li == 0:
        return {'idx': idx, 'p': np.ones(H), 'trivial': True}
    p = activity_probs(pb, H, pseudo)
    return {'idx': idx, 'p': p, 'trivial': False}

def _detrend_diff(x):
    x = np.asarray(x, dtype=float)
    return np.diff(x)

def useful_past_channel(chan, target, min_pairs=12, thresh=0.2, max_lag=4):
    dc = _detrend_diff(chan)
    dt = _detrend_diff(target)
    n = min(len(dc), len(dt))
    if n < min_pairs:
        return False
    dc, dt = (dc[-n:], dt[-n:])
    if np.std(dc) <= 0 or np.std(dt) <= 0:
        return False
    for lag in range(1, max_lag + 1):
        m = n - lag
        if m < min_pairs:
            break
        a, b = (dc[:m], dt[lag:])
        if np.std(a) <= 0 or np.std(b) <= 0:
            continue
        need = max(thresh, 2.5 / np.sqrt(m))
        if abs(float(np.corrcoef(a, b)[0, 1])) >= need:
            return True
    return False

def useful_known_channel(chan, n_past, min_var_points=4):
    c = np.asarray(chan, dtype=float)
    past, fut = (c[:n_past], c[n_past:])
    if len(past) < min_var_points or len(fut) == 0:
        return False
    return bool(np.std(past) > 0 and (np.std(fut) > 0 or abs(fut.mean() - past.mean()) > 0))

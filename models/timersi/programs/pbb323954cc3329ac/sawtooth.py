import numpy as np

def _slope(d):
    pos = d[d > 0]
    if pos.size < 100:
        return None
    s = float(np.median(pos))
    for _ in range(5):
        m = np.abs(d - s) <= abs(s) * 0.02
        if m.sum() < 50:
            return None
        s = float(np.median(d[m]))
    return s if np.isfinite(s) and s > 0 else None

def _events(x, s, clean):
    idx, Us, Ls = ([], [], [])
    n = len(x)
    for a in np.where(~clean)[0]:
        if a < 1 or a + 3 >= n - 1:
            continue
        if not clean[a - 1] or clean[a + 1] or (not clean[a + 2]):
            continue
        x0, x1, x2 = (x[a], x[a + 1], x[a + 2])
        R = x0 - x2 + 2.0 * s
        if R <= 2.0 * s:
            continue
        w = (x0 + s - x1) / R
        if not 0.0 < w < 1.0:
            continue
        U = x0 + s * (1.5 - w)
        idx.append(a)
        Us.append(U)
        Ls.append(U - R)
    return (np.asarray(idx, dtype=float), np.asarray(Us), np.asarray(Ls))

def fit_reset_ramp(x, min_cycles=25, min_clean=0.8):
    x = np.asarray(x, dtype=np.float64)
    if x.size < 400 or not np.isfinite(x).all():
        return None
    d = np.diff(x)
    s = _slope(d)
    if s is None:
        return None
    tol = max(abs(s) * 0.001, 1e-07)
    clean = np.abs(d - s) <= tol
    frac = float(clean.mean())
    if frac < min_clean:
        return None
    idx, U, L = _events(x, s, clean)
    if idx.size < min_cycles + 1:
        return None
    gap = (U[1:] - L[:-1]) / s
    span = np.diff(idx)
    good = np.abs(gap - span) < 1.5
    Lg = L[:-1][good]
    gap = gap[good]
    if gap.size < min_cycles:
        return None
    lo, hi = np.percentile(gap, [1.0, 99.0])
    gk = gap[(gap >= lo) & (gap <= hi)]
    if gk.size >= min_cycles:
        gap = gk
    lo, hi = np.percentile(L, [1.0, 99.0])
    Lk = L[(L >= lo) & (L <= hi)]
    Lb = Lk if Lk.size >= min_cycles else L
    return {'s': s, 'frac': frac, 'clean': clean, 'gap': gap, 'Lpool': Lb, 'gapL': Lg, 'gapfull': (U[1:] - L[:-1])[good] / s, 'U': U, 'L': L, 'idx': idx, 'ncyc': int(gap.size), 'period': float(np.mean(gap))}

def _last_clean_index(clean, n, lookback):
    t0 = n - 1
    lim = max(1, n - 1 - lookback)
    while t0 > lim:
        if clean[t0 - 1] and (t0 >= n - 1 or clean[t0]):
            break
        t0 -= 1
    return t0

def simulate(x, fit, H, qs, M=3000, N=4, seed=0, lookback=12, recent=None, obs_scale=0.15, jitter=0.0, gjitter=0.0, pair=False, Lshape=None):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    s = fit['s']
    gap, Lp = (fit['gap'], fit['Lpool'])
    if pair:
        gap_p, L_p = (fit['gapfull'], fit['gapL'])
    if Lshape is not None:
        Lp = np.median(Lp) + Lp.std() / (Lshape.std() + 1e-12) * (Lshape - np.median(Lshape))
    if recent is not None:
        if gap.size > recent:
            gap = gap[-recent:]
        if Lp.size > recent:
            Lp = Lp[-recent:]
    ng, nl = (gap.size, Lp.size)
    n = x.size
    t0 = _last_clean_index(fit['clean'], n, lookback)
    nobs = n - 1 - t0
    step = s / N
    v = np.full(M, x[t0] + 0.5 * s - 0.5 * step)
    anchor = None
    if fit['idx'].size:
        j = int(np.searchsorted(fit['idx'], t0 - 2, 'right')) - 1
        if j >= 0:
            elapsed = n - 1 - fit['idx'][j]
            if 0 <= elapsed <= gap.max() + 5:
                anchor = (fit['L'][j], fit['idx'][j])
    if anchor is not None:
        g = gap[rng.integers(0, ng, M)]
        thr = anchor[0] + s * g
    else:
        thr = fit['U'][rng.integers(0, fit['U'].size, M)]
    for _ in range(60):
        low = thr < v + step
        if not low.any():
            break
        k = int(low.sum())
        if anchor is not None:
            thr[low] = anchor[0] + s * gap[rng.integers(0, ng, k)]
        else:
            thr[low] = fit['U'][rng.integers(0, fit['U'].size, k)]
    low = thr < v + step
    if low.any():
        thr[low] = v[low] + step + abs(s) * rng.random(int(low.sum()))
    means = np.empty((nobs + H, M))
    for t in range(nobs + H):
        acc = np.zeros(M)
        for _ in range(N):
            v += step
            over = v > thr
            if over.any():
                k = int(over.sum())
                if pair:
                    jj = rng.integers(0, gap_p.size, k)
                    newL = L_p[jj]
                    newg = gap_p[jj]
                else:
                    newL = Lp[rng.integers(0, nl, k)]
                    newg = gap[rng.integers(0, ng, k)]
                if jitter > 0:
                    newL = newL + jitter * rng.standard_normal(k)
                if gjitter > 0:
                    newg = newg + gjitter * rng.standard_normal(k)
                v[over] = newL + (v[over] - thr[over])
                thr[over] = newL + s * newg
            acc += v
        means[t] = acc / N
    if nobs > 0:
        obs = x[t0 + 1:]
        sd = max(abs(s) * obs_scale, 1e-09)
        ll = -((means[:nobs] - obs[:, None]) ** 2).sum(0) / (2.0 * sd * sd)
        w = np.exp(ll - ll.max())
        if not np.isfinite(w).all() or w.sum() <= 1e-10:
            w = np.ones(M)
    else:
        w = np.ones(M)
    w = w / w.sum()
    ens = means[nobs:]
    out = np.empty((H, len(qs)))
    order = np.argsort(ens, axis=1)
    for h in range(H):
        o = order[h]
        ww = w[o]
        cw = np.cumsum(ww) - 0.5 * ww
        out[h] = np.interp(qs, cw, ens[h][o])
    return (out, ens, w)

def pinball(y, q, qs):
    e = y[:, None] - q
    return float(np.mean(2.0 * np.maximum(qs * e, (qs - 1.0) * e)))

def backtest_gain(x, fit, H, qs, origins=8, gap=97, M=600, N=4, window=360, **kw):
    n = x.size
    gains = []
    tol = max(abs(fit['s']) * 0.001, 1e-07)
    for i in range(origins):
        end = n - H - i * gap
        if end < max(800, window + 10):
            break
        hist = x[:end]
        truth = x[end:end + H]
        sub = dict(fit)
        sub['clean'] = np.abs(np.diff(hist) - fit['s']) <= tol
        sub['idx'] = fit['idx'][fit['idx'] < end - 4]
        sub['L'] = fit['L'][:sub['idx'].size]
        try:
            q, _, _ = simulate(hist, sub, H, qs, M=M, N=N, seed=1000 + i, **kw)
        except Exception:
            continue
        base = np.tile(np.quantile(hist[-window:], qs), (H, 1))
        gains.append(pinball(truth, base, qs) - pinball(truth, q, qs))
    if not gains:
        return (0.0, 0)
    return (float(np.mean(gains)), len(gains))

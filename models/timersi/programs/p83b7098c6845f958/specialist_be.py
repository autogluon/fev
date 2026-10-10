import numpy as np
LAG_DAYS = (1, 2, 3, 7)

def daily_matrices(price, gen, load, n_days):
    P = np.asarray(price, float).reshape(-1, 24)[-n_days:]
    G = np.asarray(gen, float).reshape(-1, 24)[-n_days:]
    S = np.asarray(load, float).reshape(-1, 24)[-n_days:]
    return (P, G, S)

def _lag(M, L):
    out = np.full_like(M, np.nan)
    if L < M.shape[0]:
        out[L:] = M[:-L]
    return out

def feature_tensor(P, G, S, dow, off, doy):
    D = P.shape[0]

    def rep(a):
        return np.repeat(np.asarray(a, float)[:, None], 24, 1)
    parts, names = ([], [])
    for L in LAG_DAYS:
        parts.append(_lag(P, L))
        names.append('p_lag%d' % L)
    p1 = _lag(P, 1)
    parts.append(np.roll(p1, 1, axis=1))
    names.append('p1_hm1')
    parts.append(np.roll(p1, -1, axis=1))
    names.append('p1_hp1')
    for arr, nm in ((p1.max(1), 'prev_max'), (p1.min(1), 'prev_min'), (p1.mean(1), 'prev_mean'), (p1[:, 23], 'prev_last'), (_lag(P, 2).mean(1), 'p2_mean'), (_lag(P, 7).mean(1), 'p7_mean')):
        parts.append(rep(arr))
        names.append(nm)
    pm = P.mean(1)
    cs = np.concatenate([[0.0], np.cumsum(np.nan_to_num(pm))])
    cw = np.full(D, np.nan)
    cw2 = np.full(D, np.nan)
    k = np.arange(14, D)
    if k.size:
        cw[14:] = (cs[14:D] - cs[7:D - 7]) / 7.0
        cw2[14:] = (cs[7:D - 7] - cs[0:D - 14]) / 7.0
    parts.append(rep(cw))
    names.append('week_mean')
    parts.append(rep(cw2))
    names.append('week_mean_prev')
    parts.append(S)
    names.append('S_h')
    parts.append(rep(S.mean(1)))
    names.append('S_mean')
    parts.append(rep(S.max(1)))
    names.append('S_max')
    parts.append(G)
    names.append('G_h')
    parts.append(rep(G.mean(1)))
    names.append('G_mean')
    marg = G - S
    parts.append(marg)
    names.append('margin_h')
    parts.append(rep(marg.min(1)))
    names.append('margin_min')
    parts.append(S / np.maximum(np.abs(S.mean(1))[:, None], 1e-06))
    names.append('S_shape')
    parts.append(S - _lag(S, 1))
    names.append('S_dh')
    parts.append(rep(S.mean(1) - _lag(S, 1).mean(1)))
    names.append('S_dday')
    parts.append(rep(G.mean(1) - _lag(G, 1).mean(1)))
    names.append('G_dday')
    for i in range(7):
        parts.append(rep((dow == i).astype(float)))
        names.append('dow%d' % i)
    parts.append(rep(off))
    names.append('off')
    for L in LAG_DAYS:
        lo = np.full(D, np.nan)
        if L < D:
            lo[L:] = off[:-L]
        parts.append(rep(lo))
        names.append('off_lag%d' % L)
    for kk in (1, 2):
        parts.append(rep(np.sin(2 * np.pi * kk * doy / 365.25)))
        names.append('sin%d' % kk)
        parts.append(rep(np.cos(2 * np.pi * kk * doy / 365.25)))
        names.append('cos%d' % kk)
    return (np.stack(parts, -1), names)

def _solve(Z, y, ridge):
    A = Z.T @ Z + ridge * np.eye(Z.shape[1])
    A[-1, -1] -= ridge
    try:
        return np.linalg.solve(A, Z.T @ y)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, Z.T @ y, rcond=None)[0]

def lear_predict(X, P, origin_day, train_days=1095, ridge=3.0, robust=False, clipx=4.0):
    start = max(16, origin_day - train_days)
    rows = np.arange(start, origin_day)
    if rows.size < 120:
        return None
    blk = P[start:origin_day]
    blk = blk[np.isfinite(blk)]
    if blk.size < 100:
        return None
    med = float(np.median(blk))
    mad = float(np.median(np.abs(blk - med))) * 1.4826 + 1e-06
    pred = np.zeros(24)
    for h in range(24):
        Xt = X[rows, h, :]
        yv = np.arcsinh((P[rows, h] - med) / mad)
        ok = np.isfinite(yv) & np.isfinite(Xt).all(1)
        if ok.sum() < 60:
            return None
        Xo, yo = (Xt[ok], yv[ok])
        if robust:
            mu = np.median(Xo, 0)
            sd = np.median(np.abs(Xo - mu), 0) * 1.4826
            bad = sd < 1e-09
            sd[bad] = Xo.std(0)[bad]
            sd[sd < 1e-09] = 1.0
            Zo = np.clip((Xo - mu) / sd, -clipx, clipx)
            zt = np.clip((X[origin_day, h, :] - mu) / sd, -clipx, clipx)
        else:
            mu = Xo.mean(0)
            sd = Xo.std(0)
            sd[sd < 1e-09] = 1.0
            Zo = (Xo - mu) / sd
            zt = (X[origin_day, h, :] - mu) / sd
        Zo = np.hstack([Zo, np.ones((Zo.shape[0], 1))])
        w = _solve(Zo, yo, ridge)
        zt = np.append(np.nan_to_num(zt, nan=0.0), 1.0)
        pred[h] = np.sinh(float(zt @ w)) * mad + med
    return pred

def rolling_oos_series(X, P, origin_day, n_days, train_days=1095, ridge=3.0, robust=False, step=28):
    first = max(40, origin_day - n_days)
    out = np.full((origin_day - first, 24), np.nan)
    b = first
    while b < origin_day:
        end = min(b + step, origin_day)
        start = max(16, b - train_days)
        rows = np.arange(start, b)
        if rows.size < 120:
            b = end
            continue
        blk = P[start:b]
        blk = blk[np.isfinite(blk)]
        med = float(np.median(blk))
        mad = float(np.median(np.abs(blk - med))) * 1.4826 + 1e-06
        tgt = np.arange(b, end)
        for h in range(24):
            Xt = X[rows, h, :]
            yv = np.arcsinh((P[rows, h] - med) / mad)
            ok = np.isfinite(yv) & np.isfinite(Xt).all(1)
            if ok.sum() < 60:
                continue
            Xo, yo = (Xt[ok], yv[ok])
            mu = Xo.mean(0)
            sd = Xo.std(0)
            sd[sd < 1e-09] = 1.0
            Zo = np.hstack([(Xo - mu) / sd, np.ones((Xo.shape[0], 1))])
            w = _solve(Zo, yo, ridge)
            Zp = np.hstack([np.nan_to_num((X[tgt, h, :] - mu) / sd, nan=0.0), np.ones((tgt.size, 1))])
            out[tgt - first, h] = np.sinh(Zp @ w) * mad + med
        b = end
    return (first, out)

import numpy as np
LAG_DAYS = (1, 2, 3, 7)

def build_daily(price, sysload, comed):
    n = len(price) // 24 * 24
    P = np.asarray(price[-n:], float).reshape(-1, 24)
    S = np.asarray(sysload[-n:], float).reshape(-1, 24)
    C = np.asarray(comed[-n:], float).reshape(-1, 24)
    return (P, S, C)

def feature_tensor(P, S, C, wd, hol_flag):
    D = len(P)

    def lag(M, L):
        out = np.full_like(M, np.nan)
        out[L:] = M[:-L]
        return out

    def rep(a):
        return np.repeat(np.asarray(a, float)[:, None], 24, 1)
    parts, names = ([], [])
    for L in LAG_DAYS:
        parts.append(lag(P, L))
        names.append('p_lag%d' % L)
    p1 = lag(P, 1)
    parts.append(np.roll(p1, 1, axis=1))
    names.append('p1_hm1')
    parts.append(np.roll(p1, -1, axis=1))
    names.append('p1_hp1')
    for arr, nm in [(p1.max(1), 'prev_max'), (p1.min(1), 'prev_min'), (p1.mean(1), 'prev_mean'), (p1[:, 23], 'prev_last'), (lag(P, 2).mean(1), 'p2_mean'), (lag(P, 7).mean(1), 'p7_mean')]:
        parts.append(rep(arr))
        names.append(nm)
    pm = P.mean(1)
    cw = np.full(D, np.nan)
    cw2 = np.full(D, np.nan)
    cs = np.concatenate([[0.0], np.cumsum(np.nan_to_num(pm))])
    for d in range(14, D):
        cw[d] = (cs[d] - cs[d - 7]) / 7.0
        cw2[d] = (cs[d - 7] - cs[d - 14]) / 7.0
    parts.append(rep(cw))
    names.append('week_mean')
    parts.append(rep(cw2))
    names.append('week_mean_prev')
    parts.append(S)
    names.append('S_h')
    parts.append(rep(S.max(1)))
    names.append('S_max')
    parts.append(rep(S.mean(1)))
    names.append('S_mean')
    parts.append(C)
    names.append('C_h')
    parts.append(rep(C.mean(1)))
    names.append('C_mean')
    parts.append(S - lag(S, 1))
    names.append('S_dh')
    parts.append(S / np.maximum(S.mean(1)[:, None], 1e-06))
    names.append('S_shape')
    parts.append(rep(S.mean(1)) - rep(lag(S, 1).mean(1)))
    names.append('S_dday')
    parts.append(C / np.maximum(np.abs(S), 1e-06))
    names.append('C_share')
    for i in range(7):
        parts.append(rep((wd == i).astype(float)))
        names.append('dow%d' % i)
    parts.append(rep(hol_flag))
    names.append('hol')
    off = ((wd >= 5) | (hol_flag > 0.5)).astype(float)
    for L in LAG_DAYS:
        lo = np.full(D, np.nan)
        lo[L:] = off[:-L]
        parts.append(rep(lo))
        names.append('off_lag%d' % L)
    parts.append(rep(off))
    names.append('off')
    doy = np.arange(D, dtype=float)
    return (np.stack(parts, -1), names)

def add_seasonal(X, doy):
    extra = []
    for k in (1, 2):
        extra.append(np.repeat(np.sin(2 * np.pi * k * doy / 365.25)[:, None], 24, 1))
        extra.append(np.repeat(np.cos(2 * np.pi * k * doy / 365.25)[:, None], 24, 1))
    return np.concatenate([X, np.stack(extra, -1)], -1)

def _fit_hour(Z, y, ridge):
    A = Z.T @ Z + ridge * np.eye(Z.shape[1])
    A[-1, -1] -= ridge
    try:
        return np.linalg.solve(A, Z.T @ y)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, Z.T @ y, rcond=None)[0]

def lear_predict(X, P, origin, train_days=1095, ridge=3.0, robust=False, clipx=4.0, return_resid=False):
    start = max(16, origin - train_days)
    rows = np.arange(start, origin)
    if len(rows) < 60:
        return None if not return_resid else (None, None)
    blk = P[start:origin]
    blk = blk[np.isfinite(blk)]
    med = np.median(blk)
    mad = np.median(np.abs(blk - med)) * 1.4826 + 1e-06
    pred = np.zeros(24)
    resid = np.zeros((len(rows), 24))
    for h in range(24):
        Xt = X[rows, h, :]
        yv = np.arcsinh((P[rows, h] - med) / mad)
        ok = np.isfinite(yv) & np.isfinite(Xt).all(1)
        if ok.sum() < 40:
            return None if not return_resid else (None, None)
        Xo, yo = (Xt[ok], yv[ok])
        if robust:
            mu = np.median(Xo, 0)
            sd = np.median(np.abs(Xo - mu), 0) * 1.4826
            bad = sd < 1e-09
            sd[bad] = Xo.std(0)[bad]
            sd[sd < 1e-09] = 1.0
            Zo = np.clip((Xo - mu) / sd, -clipx, clipx)
            zt = np.clip((X[origin, h, :] - mu) / sd, -clipx, clipx)
        else:
            mu = Xo.mean(0)
            sd = Xo.std(0)
            sd[sd < 1e-09] = 1.0
            Zo = (Xo - mu) / sd
            zt = (X[origin, h, :] - mu) / sd
        Zo = np.hstack([Zo, np.ones((len(Zo), 1))])
        w = _fit_hour(Zo, yo, ridge)
        r = np.full(len(rows), np.nan)
        r[ok] = yo - Zo @ w
        resid[:, h] = r
        zt = np.append(np.nan_to_num(zt, nan=0.0), 1.0)
        pred[h] = np.sinh(float(zt @ w)) * mad + med
    if return_resid:
        return (pred, (resid, med, mad))
    return pred

def rolling_oos_series(X, P, origin, n_days, train_days=1095, ridge=3.0, robust=False, step=28):
    first = max(40, origin - n_days)
    out = np.full((origin - first, 24), np.nan)
    b = first
    while b < origin:
        end = min(b + step, origin)
        start = max(16, b - train_days)
        rows = np.arange(start, b)
        if len(rows) < 120:
            b = end
            continue
        blk = P[start:b]
        blk = blk[np.isfinite(blk)]
        med = np.median(blk)
        mad = np.median(np.abs(blk - med)) * 1.4826 + 1e-06
        tgt = np.arange(b, end)
        for h in range(24):
            Xt = X[rows, h, :]
            yv = np.arcsinh((P[rows, h] - med) / mad)
            ok = np.isfinite(yv) & np.isfinite(Xt).all(1)
            if ok.sum() < 60:
                continue
            Xo, yo = (Xt[ok], yv[ok])
            if robust:
                mu = np.median(Xo, 0)
                sd = np.median(np.abs(Xo - mu), 0) * 1.4826
                bad = sd < 1e-09
                sd[bad] = Xo.std(0)[bad]
                sd[sd < 1e-09] = 1.0
                Zo = np.clip((Xo - mu) / sd, -4.0, 4.0)
                Zp = np.clip((X[tgt, h, :] - mu) / sd, -4.0, 4.0)
            else:
                mu = Xo.mean(0)
                sd = Xo.std(0)
                sd[sd < 1e-09] = 1.0
                Zo = (Xo - mu) / sd
                Zp = (X[tgt, h, :] - mu) / sd
            Zo = np.hstack([Zo, np.ones((len(Zo), 1))])
            w = _fit_hour(Zo, yo, ridge)
            Zp = np.hstack([np.nan_to_num(Zp, nan=0.0), np.ones((len(Zp), 1))])
            out[tgt - first, h] = np.sinh(Zp @ w) * mad + med
        b = end
    return (first, out)

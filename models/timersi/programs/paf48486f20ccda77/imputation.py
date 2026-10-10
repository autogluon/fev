import numpy as np
import pandas as pd

def _wridge(X, y, w, lam):
    Xw = X * np.sqrt(w)[:, None]
    A = Xw.T @ Xw + lam * np.eye(X.shape[1])
    return np.linalg.solve(A, Xw.T @ (y * np.sqrt(w)))

def _seasonal_init(logrow, key):
    s = pd.Series(logrow)
    prof = s.groupby(key).transform('median')
    filled = prof + (s - prof).interpolate(limit_direction='both')
    med = np.nanmedian(logrow)
    return filled.fillna(prof).fillna(med if np.isfinite(med) else 0.0).values

def _gap_distance(obs):
    L = len(obs)
    idx = np.arange(L)
    oi = idx[obs]
    if len(oi) == 0:
        return np.full(L, 1000000.0)
    left = np.searchsorted(oi, idx, 'right') - 1
    right = np.searchsorted(oi, idx, 'left')
    dl = np.where(left >= 0, idx - oi[np.clip(left, 0, len(oi) - 1)], 10 ** 6)
    dr = np.where(right < len(oi), oi[np.clip(right, 0, len(oi) - 1)] - idx, 10 ** 6)
    return np.minimum(dl, dr).astype(float)

def reconstruct(raw, observed, timestamps, met=None, halflife_h=24 * 28, lam=1.0, anchor_tau=6.0, iters=6, floor=0.001):
    raw = np.asarray(raw, float)
    D, L = raw.shape
    ts = pd.DatetimeIndex(timestamps[:L])
    hour = ts.hour.values.astype(float)
    dow = ts.dayofweek.values.astype(float)
    key = (hour * 2 + (dow >= 5)).astype(int)
    tnorm = (np.arange(L) - L) / (24 * 30.0)
    w = 0.5 ** (-tnorm * 30.0 * 24.0 / halflife_h)
    lg = np.log(np.clip(raw, floor, None))
    obs = np.asarray(observed, bool) & np.isfinite(lg)
    if obs.all():
        return raw.copy()
    extra = []
    if met is not None and len(met):
        met = np.asarray(met, float)[:, :L]
        for j in range(met.shape[0]):
            col = met[j]
            sd = col.std()
            extra.append((col - col.mean()) / (sd if sd > 1e-09 else 1.0))
    cal = [np.ones(L), tnorm]
    for kk in (1, 2, 3):
        cal += [np.sin(2 * np.pi * kk * hour / 24), np.cos(2 * np.pi * kk * hour / 24)]
    cal += [(dow >= 5).astype(float), (dow == 6).astype(float)]
    cur = np.vstack([_seasonal_init(np.where(obs[d], lg[d], np.nan), key) for d in range(D)])
    dist = np.vstack([_gap_distance(obs[d]) for d in range(D)])
    decay = np.exp(-dist / anchor_tau)
    out = cur
    for _ in range(max(1, iters)):
        nxt = cur.copy()
        for d in range(D):
            o = obs[d]
            if o.sum() < 100:
                continue
            others = [cur[j] for j in range(D) if j != d]
            cols = cal + others + [others[0] * tnorm] + extra if others else cal + extra
            X = np.column_stack(cols)
            beta = _wridge(X[o], lg[d][o], w[o], lam)
            pred = X @ beta
            res = np.where(o, lg[d] - pred, np.nan)
            anch = np.nan_to_num(pd.Series(res).interpolate(limit_direction='both').values)
            nxt[d] = np.where(o, lg[d], pred + anch * decay[d])
        cur = nxt
        out = cur
    filled = np.exp(np.clip(out, -20, 20))
    return np.where(obs, raw, filled)

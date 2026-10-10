import numpy as np
LOGSHIFT = 1.0

def to_log(x):
    return np.log(np.maximum(x, 0.0) + LOGSHIFT)

def from_log(z):
    return np.exp(z) - LOGSHIFT

def dow_from_timestamps(timestamps):
    import datetime as _dt
    out = np.empty(len(timestamps), dtype=np.int64)
    for i, t in enumerate(timestamps):
        s = str(t)[:10]
        y, m, d = (int(s[0:4]), int(s[5:7]), int(s[8:10]))
        out[i] = _dt.date(y, m, d).weekday()
    return out

def weekly_factors(z, obs, dow, min_count=3, shrink=2.0):
    fac = np.zeros(7)
    if obs.sum() < 14:
        return fac
    base = np.median(z[obs])
    for d in range(7):
        sel = obs & (dow == d)
        n = int(sel.sum())
        if n >= min_count:
            fac[d] = (np.median(z[sel]) - base) * (n / (n + shrink))
    fac -= np.mean(fac)
    return fac

def local_level(z, obs, half=21, min_pts=4):
    n = len(z)
    lev = np.full(n, np.nan)
    idx = np.where(obs)[0]
    if len(idx) == 0:
        return lev
    for t in range(n):
        lo, hi = (t - half, t + half)
        sel = idx[(idx >= lo) & (idx <= hi)]
        if len(sel) >= min_pts:
            lev[t] = np.median(z[sel])
    return lev

def fill_level(lev, z, obs, reversion_days=60.0):
    n = len(lev)
    idx = np.where(np.isfinite(lev))[0]
    longrun = np.median(z[obs]) if obs.any() else 0.0
    if len(idx) == 0:
        return np.full(n, longrun)
    t = np.arange(n)
    out = np.interp(t, idx, lev[idx])
    dist = np.full(n, np.inf)
    for i in idx:
        pass
    nearest = np.searchsorted(idx, t)
    left = np.clip(nearest - 1, 0, len(idx) - 1)
    right = np.clip(nearest, 0, len(idx) - 1)
    dist = np.minimum(np.abs(t - idx[left]), np.abs(t - idx[right])).astype(float)
    w = np.exp(-dist / max(reversion_days, 1.0))
    return w * out + (1.0 - w) * longrun

def long_gap_mask(obs, min_run):
    n = len(obs)
    out = np.zeros(n, dtype=bool)
    i = 0
    while i < n:
        if obs[i]:
            i += 1
            continue
        j = i
        while j < n and (not obs[j]):
            j += 1
        if j - i >= min_run:
            out[i:j] = True
        i = j
    return out

def reconstruct(history, observed, timestamps, min_run=5):
    y = np.asarray(history, dtype=float)
    obs = np.asarray(observed, dtype=bool)
    target = long_gap_mask(obs, min_run)
    if obs.sum() < 10 or not target.any():
        return (y, np.zeros(len(y)))
    z = to_log(y)
    dow = dow_from_timestamps(timestamps[:len(y)])
    fac = weekly_factors(z, obs, dow)
    lev = fill_level(local_level(z - fac[dow], obs), z - fac[dow], obs)
    rebuilt = from_log(lev + fac[dow])
    out = np.where(target, rebuilt, y)
    return (out, target.astype(float))

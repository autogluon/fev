import numpy as np
QL = np.arange(1, 10) / 10.0

def day_table(x, hour, dow, day_index, n_days_back, drop_thr):
    days = []
    dmax = day_index.max()
    dx = np.diff(x, prepend=x[0])
    isdrop = dx < drop_thr
    for d in range(max(0, dmax - n_days_back), dmax + 1):
        m = day_index == d
        if m.sum() != 24:
            continue
        hh = hour[m]
        if not np.array_equal(hh, np.arange(24)):
            continue
        inc = dx[m].copy()
        dr = isdrop[m]
        lv = x[m].copy()
        days.append({'daytype': int(dow[m][0] >= 5), 'drop': dr.copy(), 'inc': inc, 'level': lv, 'age': dmax - d})
    return days

def fit(x, hour, dow, day_index, n_days_back=70):
    dxa = np.diff(x)
    pos = dxa[dxa > 1e-12]
    slope = float(np.median(pos)) if len(pos) else 0.0
    amp = float(np.percentile(np.abs(dxa), 99)) if len(dxa) else 0.0
    thr = -max(2.0 * slope, 0.05 * amp, 1e-09)
    days = day_table(x, hour, dow, day_index, n_days_back, thr)
    return {'days': days, 'slope': slope, 'thr': thr}

def simulate(par, last_val, fhour, fdow, nsim, rng, half_life=18.0):
    H = len(fhour)
    days = par['days']
    if not days or H != 24 or (not np.array_equal(fhour, np.arange(24))):
        return None
    want = int(fdow[0] >= 5)
    pool = [d for d in days if d['daytype'] == want]
    if len(pool) < 3:
        pool = days
    w = np.array([0.5 ** (d['age'] / half_life) for d in pool])
    w = w / w.sum()
    pick = rng.choice(len(pool), size=nsim, p=w)
    paths = np.empty((nsim, H))
    for s in range(nsim):
        d = pool[pick[s]]
        v = last_val
        for t in range(H):
            if d['drop'][t]:
                v = d['level'][t]
            else:
                v = v + d['inc'][t]
            paths[s, t] = v
    return paths

def forecast(x, hour, dow, day_index, fhour, fdow, nsim=400, seed=0, **kw):
    par = fit(x, hour, dow, day_index, **kw)
    paths = simulate(par, float(x[-1]), fhour, fdow, nsim, np.random.default_rng(seed))
    if paths is None:
        p = np.full(len(fhour), float(x[-1]))
        return (p, np.repeat(p[:, None], 9, axis=1), 0.0)
    q = np.quantile(paths, QL, axis=0).T
    p = np.median(paths, axis=0)
    spread = float(np.mean(q[:, 7] - q[:, 1]))
    return (p, q, spread)

def forecast_mix(x, hour, dow, day_index, fhour, fdow, nsim=300, seed=0, n_days_back=120, half_lives=(8.0, 24.0)):
    par = fit(x, hour, dow, day_index, n_days_back=n_days_back)
    rng = np.random.default_rng(seed)
    chunks = []
    for hl in half_lives:
        p = simulate(par, float(x[-1]), fhour, fdow, nsim, rng, half_life=hl)
        if p is not None:
            chunks.append(p)
    if not chunks:
        p = np.full(len(fhour), float(x[-1]))
        return (p, np.repeat(p[:, None], 9, axis=1), 0.0)
    paths = np.concatenate(chunks, axis=0)
    q = np.quantile(paths, QL, axis=0).T
    return (q[:, 4].copy(), q, float(np.mean(q[:, 7] - q[:, 1])))

def pooled_profile(pool, w):
    inc = np.array([d['inc'] for d in pool])
    drp = np.array([d['drop'] for d in pool])
    lvl = np.array([d['level'] for d in pool])
    order = np.argsort(-w)
    inc_p = np.empty(24)
    lvl_p = np.empty(24)
    for t in range(24):
        ok = ~drp[:, t]
        inc_p[t] = _wmed(inc[ok, t], w[ok]) if ok.any() else 0.0
        hit = drp[:, t]
        lvl_p[t] = _wmed(lvl[hit, t], w[hit]) if hit.any() else np.nan
    anyhit = drp.any(axis=0)
    if drp.any():
        allv = lvl[drp]
        allw = np.repeat(w[:, None], 24, axis=1)[drp]
        fill = _wmed(allv, allw)
    else:
        fill = np.nan
    lvl_p = np.where(np.isfinite(lvl_p), lvl_p, fill)
    return (inc_p, lvl_p)

def _wmed(v, w):
    if len(v) == 0:
        return np.nan
    o = np.argsort(v)
    v, w = (v[o], w[o])
    c = np.cumsum(w)
    if c[-1] <= 0:
        return float(np.median(v))
    return float(v[np.searchsorted(c, 0.5 * c[-1])])

def forecast_pooled(x, hour, dow, day_index, fhour, fdow, nsim=400, seed=0, n_days_back=120, half_lives=(4.0, 8.0, 16.0), shape_w=1.0):
    par = fit(x, hour, dow, day_index, n_days_back=n_days_back)
    days = par['days']
    H = len(fhour)
    if not days or H != 24 or (not np.array_equal(fhour, np.arange(24))):
        p = np.full(H, float(x[-1]))
        return (p, np.repeat(p[:, None], 9, axis=1), 0.0)
    want = int(fdow[0] >= 5)
    pool = [d for d in days if d['daytype'] == want]
    if len(pool) < 3:
        pool = days
    age = np.array([d['age'] for d in pool], float)
    rng = np.random.default_rng(seed)
    paths = []
    for hl in half_lives:
        w = 0.5 ** (age / hl)
        w = w / w.sum()
        inc_p, lvl_p = pooled_profile(pool, w)
        pick = rng.choice(len(pool), size=nsim, p=w)
        P = np.empty((nsim, H))
        for s in range(nsim):
            d = pool[pick[s]]
            v = float(x[-1])
            for t in range(H):
                if d['drop'][t]:
                    lv = lvl_p[t] if np.isfinite(lvl_p[t]) else d['level'][t]
                    v = (1.0 - shape_w) * d['level'][t] + shape_w * lv
                else:
                    iv = inc_p[t] if np.isfinite(inc_p[t]) else d['inc'][t]
                    v = v + (1.0 - shape_w) * d['inc'][t] + shape_w * iv
                P[s, t] = v
        paths.append(P)
    paths = np.concatenate(paths, axis=0)
    q = np.quantile(paths, QL, axis=0).T
    return (q[:, 4].copy(), q, float(np.mean(q[:, 7] - q[:, 1])))

def native_drop_hour(point, last_val, tol=0.25):
    full = np.concatenate([[last_val], np.asarray(point, float)])
    dx = np.diff(full)
    j = int(np.argmin(dx))
    up = dx[dx > 0]
    typ = float(np.median(up)) if len(up) else 0.0
    rng = float(np.max(full) - np.min(full))
    if dx[j] < -max(2.0 * typ, tol * rng, 1e-09):
        return j
    return -1

def forecast_guided(x, hour, dow, day_index, fhour, fdow, guide=-1, boost=3.0, nsim=400, seed=0, n_days_back=120, half_lives=(4.0, 8.0, 16.0)):
    par = fit(x, hour, dow, day_index, n_days_back=n_days_back)
    days = par['days']
    H = len(fhour)
    if not days or H != 24 or (not np.array_equal(fhour, np.arange(24))):
        p = np.full(H, float(x[-1]))
        return (p, np.repeat(p[:, None], 9, axis=1), 0.0)
    want = int(fdow[0] >= 5)
    pool = [d for d in days if d['daytype'] == want]
    if len(pool) < 3:
        pool = days
    age = np.array([d['age'] for d in pool], float)
    first = np.array([int(np.argmax(d['drop'])) if d['drop'].any() else -1 for d in pool])
    agree = first == guide
    rng = np.random.default_rng(seed)
    chunks = []
    for hl in half_lives:
        w = 0.5 ** (age / hl)
        if agree.any() and (not agree.all()):
            w = w * np.where(agree, boost, 1.0)
        w = w / w.sum()
        pick = rng.choice(len(pool), size=nsim, p=w)
        P = np.empty((nsim, H))
        for s in range(nsim):
            d = pool[pick[s]]
            v = float(x[-1])
            for t in range(H):
                v = d['level'][t] if d['drop'][t] else v + d['inc'][t]
                P[s, t] = v
        chunks.append(P)
    paths = np.concatenate(chunks, axis=0)
    q = np.quantile(paths, QL, axis=0).T
    return (q[:, 4].copy(), q, float(np.mean(q[:, 7] - q[:, 1])))

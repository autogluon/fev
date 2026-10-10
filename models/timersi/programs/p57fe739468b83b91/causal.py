import numpy as np
from calendarx import day_type, temp_channels
MAX_FIT_HOURS = 26280
FULL_MIN = 13000

def _design(ts, T, t_ref, full):
    n = ts.size
    hour = (ts.astype('datetime64[h]').astype('int64') % 24).astype(int)
    d = ts.astype('datetime64[D]')
    dow = ((d.astype('int64') + 3) % 7).astype(int)
    dt = day_type(ts)
    ma24, hdd, cdd, _ = temp_channels(T)
    Tn = (np.asarray(T, float) - 60.0) / 25.0
    Mn = (ma24 - 60.0) / 25.0
    cols = [np.ones(n)]

    def oh(idx, k):
        Z = np.zeros((n, k))
        Z[np.arange(n), idx] = 1.0
        return Z

    def byhour(v):
        Z = np.zeros((n, 24))
        Z[np.arange(n), hour] = v
        return Z
    if full:
        year = d.astype('datetime64[Y]').astype(int) + 1970
        doy = (d - d.astype('datetime64[Y]')).astype(int) + 1
        month = (d.astype('datetime64[M]').astype('int64') % 12).astype(int)
        trend = (ts.astype('datetime64[h]').astype('int64') - t_ref) / 8760.0
        cols += [trend[:, None], (trend ** 2)[:, None]]
        cols.append(oh(hour * 7 + dow, 168)[:, 1:])
        cols.append(oh(month, 12)[:, 1:])
        cols.append(byhour(dt))
        for h in (1, 2, 3):
            cols += [np.sin(2 * np.pi * h * doy / 365.25)[:, None], np.cos(2 * np.pi * h * doy / 365.25)[:, None]]
        for v in (Tn, Tn ** 2, Tn ** 3):
            cols += [v[:, None], oh(month, 12) * v[:, None], byhour(v)]
        for v in (Mn, Mn ** 2, Mn ** 3):
            cols += [v[:, None], byhour(v)]
        cols += [hdd[:, None], cdd[:, None], byhour(hdd), byhour(cdd)]
        _ = year
    else:
        cols.append(oh(hour, 24)[:, 1:])
        cols.append(np.column_stack([(dt == 0.5).astype(float), (dt == 1.0).astype(float)]))
        for v in (Tn, Tn ** 2, Tn ** 3, Mn, Mn ** 2):
            cols.append(v[:, None])
        cols.append(byhour(Tn))
        cols += [hdd[:, None], cdd[:, None]]
    return np.hstack([c if c.ndim == 2 else c[:, None] for c in cols])

def causal_estimate(ts_all, T_all, y_hist, obs_hist, L, start=0, lam=2.0):
    ts_all = np.asarray(ts_all, dtype='datetime64[h]')
    T_all = np.asarray(T_all, dtype=float)
    y = np.asarray(y_hist, dtype=float)
    ok = np.asarray(obs_hist, dtype=bool) & np.isfinite(y) & (y > 0)
    ok[:int(start)] = False
    n_av = int(ok[max(0, L - MAX_FIT_HOURS):L].sum())
    if n_av < 4000:
        return None
    full = n_av >= FULL_MIN
    t_ref = int(ts_all[L - 1].astype('datetime64[h]').astype('int64'))
    X = _design(ts_all, T_all, t_ref, full)
    lo = max(0, L - MAX_FIT_HOURS)
    sel = np.arange(lo, L)[ok[lo:L]]
    A = X[sel]
    b = np.log(y[sel])
    sd = A.std(0)
    sd[sd < 1e-09] = 1.0
    An = A / sd
    w = np.ones(sel.size)
    beta = None
    for _ in range(2):
        Aw = An * w[:, None]
        G = Aw.T @ Aw + lam * np.eye(An.shape[1])
        try:
            beta = np.linalg.solve(G, Aw.T @ (b * w))
        except np.linalg.LinAlgError:
            return None
        r = b - An @ beta
        s_rob = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-09
        w = (np.abs(r) <= 3.0 * s_rob).astype(float)
        if w.sum() < 0.5 * sel.size:
            w = np.ones(sel.size)
            break
    fit = X / sd @ beta
    resid = (b - An @ beta)[w > 0]
    if not np.isfinite(fit).all() or np.std(resid) > 1.0:
        return None
    est = np.exp(np.clip(fit, np.log(y[ok].min()) - 2.0, np.log(y[ok].max()) + 2.0))
    return (est, float(np.std(resid)))

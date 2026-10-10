import numpy as np
from lear import LEAR, make_features, holiday_flag
HOURS = 24
MERIT_WINDOW = 8760

def day_grid(view, card):
    L = int(view['cutoff_index'])
    H = int(view['horizon'])
    y = np.asarray(view['target_history'], float)
    if y.shape[0] != 1 or H != HOURS or L % HOURS != 0 or (L < 24 * 400):
        return None
    ts = np.asarray(view['timestamps'], dtype='datetime64[h]')
    if ts.shape[0] != L + H:
        return None
    step = (ts[1:] - ts[:-1]).astype(int)
    if not np.all(step == 1):
        return None
    if int(ts[L - 1].astype('datetime64[h]').astype(int) % 24) != 23:
        return None
    obs = np.asarray(view['target_observed'], bool)[0]
    if not obs.all():
        y = y[0].copy()
        idx = np.arange(L)
        good = obs
        if good.sum() < 24 * 300:
            return None
        y = np.interp(idx, idx[good], y[good])
    else:
        y = y[0].copy()
    known = np.asarray(view['known_features'], float) if len(view['known_features']) else None
    if known is None or known.shape[1] != L + H:
        return None
    nd = L // HOURS
    P = y.reshape(nd, HOURS)
    EX = [k[:L + H].reshape(nd + 1, HOURS) for k in known]
    if not np.isfinite(P).all() or not all((np.isfinite(x).all() for x in EX)):
        return None
    days = ts.reshape(nd + 1, HOURS)[:, 0].astype('datetime64[D]')
    dow = (days.astype(int) + 3) % 7
    hol = holiday_flag(days)
    Pfull = np.vstack([P, np.full((1, HOURS), np.nan)])
    return dict(P=Pfull, EX=EX, dow=dow, hol=hol, nd=nd, days=days)

def build_features(g):
    nd = g['nd']
    rows = [make_features(g['P'], g['EX'], g['dow'], g['hol'], D) for D in range(7, nd + 1)]
    p = len(rows[0])
    F = np.full((nd + 1, p), np.nan)
    F[7:] = np.stack(rows)
    return F

def residual_quantiles(F, P, D, levels, holdout=120, window=1092, max_iter=250):
    end = D - holdout
    lo = max(7, end - window)
    if end - lo < 120 or holdout < 40:
        return (None, None)
    mdl = LEAR(criterion='aic')
    pred = mdl.static_fit_predict(F, P, lo, end, np.arange(end, D), max_iter=max_iter)
    if pred is None:
        return (None, None)
    resid = P[end:D] - pred
    s = np.median(np.abs(resid - np.median(resid, axis=0)), axis=0) * 1.4826
    s = np.where(s < 1e-06, np.median(s) + 1e-06, s)
    z = (resid / s).ravel()
    z = z[np.isfinite(z)]
    if z.size < 200:
        return (None, None)
    qz = np.quantile(z, levels)
    return (qz, s)

def constructed_known(view, g):
    L = int(view['cutoff_index'])
    H = int(view['horizon'])
    known = np.asarray(view['known_features'], float)
    names = list(view['known_names'])
    extra = []
    enames = []
    li = next((i for i, n in enumerate(names) if 'Load' in n), None)
    pi = next((i for i, n in enumerate(names) if 'PV' in n or 'Wind' in n), None)
    if li is not None and pi is not None:
        extra.append(known[li] - known[pi])
        enames.append('residual_load')
    ts = np.asarray(view['timestamps'], dtype='datetime64[h]')[:L + H]
    days = ts.astype('datetime64[D]')
    dow = (days.astype(int) + 3) % 7
    hol = holiday_flag(np.unique(days))
    hmap = {d: h for d, h in zip(np.unique(days), hol)}
    hflag = np.array([hmap[d] for d in days], float)
    wend = (dow >= 5).astype(float)
    extra.append(np.maximum(hflag, wend))
    enames.append('non_working_day')
    if li is not None and pi is not None:
        y = np.asarray(view['target_history'], float)[0]
        try:
            pr = merit_order_proxy(y, known[li] - known[pi], L, window=MERIT_WINDOW)
        except Exception:
            pr = None
        if pr is not None and np.isfinite(pr).all():
            extra.append(pr)
            enames.append('merit_order_price_proxy')
    if not extra:
        return (known, names)
    return (np.vstack([known] + [e[None, :] for e in extra]), names + enames)

def merit_order_proxy(y_past, rl, L, window=8760, nbins=40):
    from sklearn.isotonic import IsotonicRegression
    lo = max(0, L - window)
    x = rl[lo:L]
    yy = y_past[lo:L]
    if x.size < 2000:
        return None
    edges = np.quantile(x, np.linspace(0, 1, nbins + 1))
    edges = np.unique(edges)
    if edges.size < 8:
        return None
    idx = np.clip(np.digitize(x, edges[1:-1]), 0, edges.size - 2)
    cx = np.zeros(edges.size - 1)
    cy = np.zeros(edges.size - 1)
    cnt = np.zeros(edges.size - 1)
    for b in range(edges.size - 1):
        m = idx == b
        if m.sum() >= 5:
            cx[b] = np.median(x[m])
            cy[b] = np.median(yy[m])
            cnt[b] = m.sum()
    keep = cnt > 0
    if keep.sum() < 6:
        return None
    cx, cy, cnt = (cx[keep], cy[keep], cnt[keep])
    iso = IsotonicRegression(increasing=True, out_of_bounds='clip')
    iso.fit(cx, cy, sample_weight=cnt)
    return iso.predict(np.clip(rl, cx[0], cx[-1]))

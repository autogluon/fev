import numpy as np

def _dates(iso_strings):
    return np.array(iso_strings, dtype='datetime64[D]')

def _dow(d):
    return ((d.astype('int64') + 3) % 7).astype(int)

def _ymd(d):
    y = d.astype('datetime64[Y]').astype(int) + 1970
    m = d.astype('datetime64[M]').astype(int) % 12 + 1
    day = (d - d.astype('datetime64[M]')).astype(int) + 1
    return (y, m, day)

def _rolling_median(x, win):
    n = len(x)
    out = np.full(n, np.nan)
    half = win // 2
    for i in range(n):
        lo, hi = (max(0, i - half), min(n, i + half + 1))
        seg = x[lo:hi]
        seg = seg[np.isfinite(seg)]
        if len(seg) >= 8:
            out[i] = np.median(seg)
    idx = np.where(np.isfinite(out))[0]
    if len(idx) == 0:
        return out
    out[:idx[0]] = out[idx[0]]
    out[idx[-1] + 1:] = out[idx[-1]]
    bad = ~np.isfinite(out)
    if bad.any():
        out[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), out[~bad])
    return out

def _grouped_median(keys, values, shrink=6.0, center=True):
    eff = {}
    for key in np.unique(keys):
        sel = values[keys == key]
        sel = sel[np.isfinite(sel)]
        if len(sel) == 0:
            continue
        eff[key] = float(np.median(sel)) * (len(sel) / (len(sel) + shrink))
    if center and eff:
        mid = float(np.median(list(eff.values())))
        eff = {k: v - mid for k, v in eff.items()}
    return eff

def fit(history, timestamps, cutoff, horizon, holiday=None, start=0, min_points=90):
    y = np.asarray(history, float)[start:cutoff]
    dates = _dates(timestamps)[start:cutoff + horizon]
    past_d, fut_d = (dates[:len(y)], dates[len(y):len(y) + horizon])
    hol = np.asarray(holiday, float)[start:cutoff + horizon] if holiday is not None else np.zeros(len(y) + horizon)
    hol_past, hol_fut = (hol[:len(y)], hol[len(y):len(y) + horizon])
    pos = y > 0
    if pos.sum() < min_points:
        return None
    ly = np.where(pos, np.log(np.maximum(y, 1e-06)), np.nan)
    base = _rolling_median(ly, 29)
    resid = ly - base
    dow_p, dom_p = (_dow(past_d), _ymd(past_d)[2])
    recent = np.zeros(len(y), bool)
    recent[-364:] = True
    dowf = _grouped_median(dow_p[recent], resid[recent], shrink=2.0)
    r2 = resid - np.array([dowf.get(d, 0.0) for d in dow_p])
    domf = _grouped_median(dom_p, r2, shrink=6.0)
    r3 = r2 - np.array([domf.get(d, 0.0) for d in dom_p])
    holf = {}
    for code in np.unique(hol_past[hol_past != 0]):
        sel = r3[hol_past == code]
        sel = sel[np.isfinite(sel)]
        if len(sel) >= 2:
            holf[float(code)] = float(np.median(sel)) * (len(sel) / (len(sel) + 1.0))
    seasonal = np.array([dowf.get(d, 0.0) for d in dow_p]) + np.array([domf.get(d, 0.0) for d in dom_p]) + np.array([holf.get(float(c), 0.0) for c in hol_past])
    des = ly - seasonal
    tail = des[-28:]
    tail = tail[np.isfinite(tail)]
    if len(tail) < 10:
        return None
    lev = float(np.median(tail))
    prev = des[-84:-28]
    prev = prev[np.isfinite(prev)]
    slope = (lev - float(np.median(prev))) / 56.0 if len(prev) >= 20 else 0.0
    slope = float(np.clip(slope, -0.0015, 0.0015))
    yearly = {}
    if len(des) > 400:
        local = _rolling_median(des, 61)
        anom = des - local
        y_, m_, d_ = _ymd(past_d)
        for mm, dd, val in zip(m_, d_, anom):
            if np.isfinite(val):
                yearly.setdefault((int(mm), int((dd - 1) // 7)), []).append(val)
    yrf = {k: float(np.median(v)) for k, v in yearly.items() if len(v) >= 2}
    dow_f, dom_f = (_dow(fut_d), _ymd(fut_d)[2])
    mon_f = _ymd(fut_d)[1]
    scatter = r3[-364:]
    scatter = scatter[np.isfinite(scatter)]
    return dict(dowf=dowf, domf=domf, holf=holf, yrf=yrf, lev=lev, slope=slope, dow_f=dow_f, dom_f=dom_f, mon_f=mon_f, hol_fut=hol_fut, noise=float(np.std(scatter)) if len(scatter) > 30 else np.nan)

def predict(model, damp_year=0.5, damp_trend=1.0):
    h = len(model['dow_f'])
    out = np.empty(h)
    for i in range(h):
        eff = model['dowf'].get(model['dow_f'][i], 0.0) + model['domf'].get(model['dom_f'][i], 0.0)
        code = float(model['hol_fut'][i])
        if code != 0:
            eff += model['holf'].get(code, 0.0)
        eff += damp_year * model['yrf'].get((int(model['mon_f'][i]), int((model['dom_f'][i] - 1) // 7)), 0.0)
        out[i] = np.exp(model['lev'] + damp_trend * model['slope'] * (i + 14) + eff)
    return out

import numpy as np

def _tod_doy(timestamps):
    ts = np.asarray(timestamps).astype('datetime64[m]')
    day = ts.astype('datetime64[D]')
    minutes = (ts - day).astype('timedelta64[m]').astype(int)
    tod = (minutes // 15).astype(int)
    doy = (day - day.astype('datetime64[Y]')).astype(int)
    dayidx = (day - day[0]).astype(int)
    return (tod, doy.astype(int), dayidx.astype(int))

def calendar(view):
    tod, doy, dayidx = _tod_doy(view['timestamps'])
    kn = list(view['known_names'])
    if 'day_length' in kn:
        dl = np.asarray(view['known_features'][kn.index('day_length')], dtype=float)
    else:
        dl = 735.0 + 285.0 * np.cos(2 * np.pi * (doy - 172) / 365.25)
    if not np.all(np.isfinite(dl)):
        fill = np.nanmedian(dl[np.isfinite(dl)]) if np.any(np.isfinite(dl)) else 735.0
        dl = np.where(np.isfinite(dl), dl, fill)
    return dict(tod=tod, doy=doy, dayidx=dayidx, dl=dl)

def _grid(dlv, todv, yv, levels, q):
    table = np.zeros((len(levels), 96))
    for i, lev in enumerate(levels):
        m = dlv == lev
        sub_t = todv[m]
        sub_y = yv[m]
        for t in range(96):
            vals = sub_y[sub_t == t]
            if vals.size:
                table[i, t] = np.max(vals) if q is None else np.quantile(vals, q)
    return table

def solar_structure(view, env_q=0.98, dl_tol=20.0, margin=1):
    cal = calendar(view)
    L = int(view['cutoff_index'])
    y = np.asarray(view['target_history'][0], dtype=float)
    obs = np.asarray(view['target_observed'][0], dtype=bool)
    valid = obs & np.isfinite(y)
    dl, tod = (cal['dl'], cal['tod'])
    dlv, todv, yv = (dl[:L][valid], tod[:L][valid], y[valid])
    levels = np.unique(dlv)
    if levels.size == 0:
        levels = np.unique(dl)
        qt = np.zeros((len(levels), 96))
        mx = np.zeros((len(levels), 96))
    else:
        qt = _grid(dlv, todv, yv, levels, env_q)
        mx = _grid(dlv, todv, yv, levels, None)
    env = np.zeros_like(qt)
    lo_hi = np.zeros((len(levels), 2))
    for i, lev in enumerate(levels):
        sel = np.abs(levels - lev) <= dl_tol
        env[i] = qt[sel].max(0)
        mrow = mx[sel].max(0)
        nz = np.flatnonzero(mrow > 0)
        lo_hi[i] = (nz.min() - margin, nz.max() + margin) if nz.size else (0, 95)
    pad = np.concatenate([env[:, :1], env, env[:, -1:]], axis=1)
    env = np.maximum((pad[:, :-2] + pad[:, 1:-1] + pad[:, 2:]) / 3.0, 0.0)
    idx = np.clip(np.searchsorted(levels, dl), 0, len(levels) - 1)
    left = np.clip(idx - 1, 0, len(levels) - 1)
    idx = np.where(np.abs(levels[left] - dl) < np.abs(levels[idx] - dl), left, idx)
    lo, hi = (lo_hi[idx, 0], lo_hi[idx, 1])
    dark = ~((tod >= lo) & (tod <= hi))
    cs = np.where(dark, 0.0, env[idx, tod])
    mid = 0.5 * (lo + hi)
    half = np.maximum(0.5 * (hi - lo), 1.0)
    elev = np.where(dark, 0.0, np.clip(1.0 - np.abs(tod - mid) / half, 0.0, 1.0))
    day_peak = np.maximum(env.max(axis=1)[idx], 1.0)
    out = dict(cs=cs, dark=dark, elev=elev, day_peak=day_peak, cs_rel=np.clip(cs / day_peak, 0.0, 1.5))
    out.update(cal)
    return out

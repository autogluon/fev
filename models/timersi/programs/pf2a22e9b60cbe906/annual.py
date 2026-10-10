import datetime as _dt
import numpy as np
from calendar_ops import parse_day, rolling_centered_median
DOY_WINDOW = 10
MIN_LAG = 26
SHRINK = 2.0
CLIP = 0.22

def _doy(d):
    return (d - _dt.date(d.year, 1, 1)).days

def annual_profile(y, obs, ts_past, ts_future, movable_log=None):
    H = len(ts_future)
    out = np.zeros(H)
    cnt = np.zeros(H)
    L = len(ts_past)
    if L < MIN_LAG + 8:
        return (out, cnt)
    v = np.asarray(y, float).copy()
    v[~np.asarray(obs, bool)] = np.nan
    v[~(v > 0)] = np.nan
    lg = np.log(v)
    if np.isfinite(lg).sum() < MIN_LAG:
        return (out, cnt)
    resid = lg - rolling_centered_median(lg, 13)
    if movable_log is not None:
        resid = resid - np.asarray(movable_log, float)[:L]
    past_doy = np.array([_doy(parse_day(t)) for t in ts_past])
    for j, tf in enumerate(ts_future):
        tgt = _doy(parse_day(tf))
        lag = L + j - np.arange(L)
        diff = np.abs(past_doy - tgt)
        diff = np.minimum(diff, 365 - diff)
        sel = (diff <= DOY_WINDOW) & (lag >= MIN_LAG) & np.isfinite(resid)
        vals = resid[sel]
        if vals.size:
            cnt[j] = vals.size
            out[j] = np.median(vals) * (vals.size / (vals.size + SHRINK))
    return (np.clip(out, -CLIP, CLIP), cnt)

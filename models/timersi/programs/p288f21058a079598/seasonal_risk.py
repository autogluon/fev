import numpy as np
MIN_HISTORY = 60
MIN_WEEKS = 30
DEV_LO = 0.06
DEV_HI = 0.14

def _iso_week(stamps):
    out = []
    for s in np.asarray(stamps).ravel():
        try:
            out.append(int(np.datetime64(str(s)[:10], 'D').astype('datetime64[D]').astype('O').isocalendar()[1]))
        except Exception:
            out.append(0)
    return np.asarray(out, dtype=int)

def profile(history, stamps_past):
    y = np.asarray(history, dtype=float)
    if y.size < MIN_HISTORY:
        return None
    ly = np.log(np.maximum(y, 1e-06))
    n = ly.size
    base = np.empty(n)
    for i in range(n):
        a, b = (max(0, i - 6), min(n, i + 7))
        base[i] = np.median(ly[a:b])
    dev = ly - base
    w = _iso_week(stamps_past)
    if w.size != n:
        return None
    prof = {}
    for k in range(1, 54):
        v = dev[w == k]
        v = v[np.isfinite(v)]
        if v.size:
            prof[k] = float(np.median(v))
    if len(prof) < MIN_WEEKS:
        return None
    sm = {}
    for k in prof:
        nb = [prof[j] for j in ((k - 2) % 53 + 1, k, k % 53 + 1) if j in prof]
        sm[k] = float(np.median(nb))
    return sm

def risk(history, stamps_past, stamps_future):
    prof = profile(history, stamps_past)
    if not prof:
        return (0.0, None)
    fw = _iso_week(stamps_future)
    vals = [abs(prof[k]) for k in fw if k in prof]
    if not vals:
        return (0.0, prof)
    return (float(np.clip((max(vals) - DEV_LO) / (DEV_HI - DEV_LO), 0.0, 1.0)), prof)

import numpy as np
MIN_LIVE_NATIVE = 4
_RAW_PROF = (1.22, 1.48, 1.28, 1.24, 1.08, 1.04, 0.98, 1.04)
PROFILE = tuple((1.0 + 0.5 * (p - 1.0) for p in _RAW_PROF))
COLD_FAN = np.array([0.6, 0.72, 0.82, 0.91, 1.0, 1.1, 1.25, 1.45, 1.85])

def profile(week_since_open):
    if week_since_open < 1:
        return 1.0
    if week_since_open <= len(PROFILE):
        return PROFILE[week_since_open - 1]
    return 1.0

def find_start(hist, obs):
    h = np.asarray(hist, float)
    o = np.asarray(obs, bool)
    pos = (h > 0) & o & np.isfinite(h)
    anypos = pos.any(axis=0)
    if not anypos.any():
        return 0
    return int(np.argmax(anypos))

def cold_start_forecast(hist_d, obs_d, start, L, H):
    y = np.asarray(hist_d, float)[start:L]
    o = np.asarray(obs_d, bool)[start:L]
    vals = [(i + 1, float(v)) for i, (v, ok) in enumerate(zip(y, o)) if ok and np.isfinite(v) and (v > 0)]
    if not vals:
        return None
    live = L - start
    steady = float(np.median([v / profile(w) for w, v in vals]))
    point = np.array([steady * profile(live + 1 + h) for h in range(H)])
    quant = point[:, None] * COLD_FAN[None, :]
    return (point, np.sort(quant, axis=1))
MIN_TRAIL = 2
OPEN_FAN = np.array([0.72, 0.82, 0.89, 0.95, 1.0, 1.05, 1.11, 1.19, 1.33])
_TAUS = np.arange(1, 10) / 10.0

def trailing_zero_run(hist_d, obs_d, L):
    h = np.asarray(hist_d, float)[:L]
    o = np.asarray(obs_d, bool)[:L]
    run = 0
    for t in range(L - 1, -1, -1):
        if not o[t]:
            continue
        if np.isfinite(h[t]) and h[t] == 0:
            run += 1
        else:
            break
    return run

def zero_persistence(hist_d, obs_d, L, run):
    h = np.asarray(hist_d, float)[:L]
    o = np.asarray(obs_d, bool)[:L]
    keep = o & np.isfinite(h)
    seq = h[keep]
    z = seq == 0
    n00 = int(np.sum(z[:-1] & z[1:]))
    n0x = int(np.sum(z[:-1]))
    return (run + n00 + 1.0) / (run + n0x + 2.0)

def closure_mixture(hist_d, obs_d, L, H):
    run = trailing_zero_run(hist_d, obs_d, L)
    if run < MIN_TRAIL:
        return None
    h = np.asarray(hist_d, float)[:L]
    o = np.asarray(obs_d, bool)[:L]
    pos = np.where(o & np.isfinite(h) & (h > 0))[0]
    if pos.size < 4:
        return None
    level = float(np.median(h[pos[-4:]]))
    p_stay = zero_persistence(hist_d, obs_d, L, run)
    openq = level * OPEN_FAN
    point = np.empty(H)
    quant = np.empty((H, 9))
    p_closed = 1.0
    for t in range(H):
        p_closed *= p_stay
        point[t] = (1.0 - p_closed) * level
        for j, tau in enumerate(_TAUS):
            if tau <= p_closed:
                quant[t, j] = 0.0
            else:
                tau2 = (tau - p_closed) / (1.0 - p_closed)
                quant[t, j] = float(np.interp(tau2, _TAUS, openq))
    return (point, np.sort(quant, axis=1), {'run': int(run), 'p_stay': round(p_stay, 3), 'level': level})

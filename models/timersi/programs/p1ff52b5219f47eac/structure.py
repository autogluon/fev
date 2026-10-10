import numpy as np
PERIOD = 52

def _eps(y):
    return 0.05 * max(float(np.mean(np.abs(y))), 1e-12)

def circular_smooth(base, half_window):
    p = len(base)
    if half_window <= 0 or p < 3:
        return np.asarray(base, float).copy()
    k = int(min(half_window, (p - 1) // 2))
    ext = np.concatenate([base[-k:], base, base[:k]])
    ker = np.ones(2 * k + 1) / (2 * k + 1)
    return np.convolve(ext, ker, mode='valid')

def _backbone(ly, period):
    L = len(ly)
    if L >= period + 13:
        w = period if period % 2 == 1 else period + 1
        k = w // 2
        ext = np.concatenate([np.full(k, ly[:k + 1].mean()), ly, np.full(k, ly[-(k + 1):].mean())])
        m = np.convolve(ext, np.ones(w) / w, mode='valid')[:L]
        return m
    t = np.arange(L, dtype=float)
    A = np.vstack([np.ones(L), (t - t.mean()) / period]).T
    coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
    return A @ coef

def _phase(total, L, period):
    return (np.arange(total) - (L - 1)) % period

def _profile(detr, phase_past, period, half_window, rho=0.55):
    L = len(detr)
    age = (L - 1 - np.arange(L)) // period
    w = rho ** age
    num = np.zeros(period)
    den = np.zeros(period)
    np.add.at(num, phase_past, w * detr)
    np.add.at(den, phase_past, w)
    filled = den > 0
    prof = np.where(filled, num / np.maximum(den, 1e-12), 0.0)
    if not filled.all():
        idx = np.arange(period)
        good = idx[filled]
        if len(good):
            prof[~filled] = np.interp(idx[~filled], good, prof[good], period=period)
    prof = circular_smooth(prof, half_window)
    return prof - prof.mean()

def _reliability(detr, phase_past, prof, period, half_window):
    L = len(detr)
    ncyc = L / float(period)
    if ncyc >= 1.95:
        recent = np.arange(L) >= L - period
        pa = _profile(detr[recent], phase_past[recent], period, half_window, rho=1.0)
        pb = _profile(detr[~recent], phase_past[~recent], period, half_window, rho=0.55)
        va, vb = (float(pa.var()), float(pb.var()))
        if va <= 1e-12 or vb <= 1e-12:
            return (0.0, ncyc)
        c = float(np.mean(pa * pb))
        return (float(np.clip(c / max(va, vb), 0.0, 1.0)), ncyc)
    resid = detr - prof[phase_past]
    sig = float(prof.var())
    k = 2 * half_window + 1
    noi = float(resid.var()) / k
    if sig + noi <= 1e-12:
        return (0.0, ncyc)
    return (float(np.clip(sig / (sig + noi), 0.0, 1.0)), ncyc)

def _slope(ly, period, cap, var_inflation=4.0):
    L = len(ly)
    if L < 8:
        return 0.0
    t = (np.arange(L) - (L - 1) / 2.0) / float(period)
    A = np.vstack([np.ones(L), t]).T
    coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
    resid = ly - A @ coef
    s2 = float(np.sum(resid ** 2)) / max(L - 2, 1)
    den = float(np.sum(t ** 2))
    if den <= 0:
        return 0.0
    se2 = var_inflation * s2 / den
    b = float(coef[1])
    shrink = b * b / (b * b + se2) if b * b + se2 > 0 else 0.0
    return float(np.clip(b * shrink, -cap, cap))

def _trend_credibility(ly, period, cap):
    L = len(ly)
    ratios = []
    tail = period // 4
    for back in range(0, 3):
        cut = L - period * (back + 1)
        if cut < period // 2 + tail:
            break
        b = _slope(ly[max(0, cut - 2 * period):cut], period, cap)
        end = cut + period
        if end > L:
            break
        if abs(b) < 0.0001:
            ratios.append(1.0)
            continue
        before = float(np.mean(ly[cut - tail:cut]))
        after = float(np.mean(ly[end - tail:end]))
        ratios.append(float(np.clip((after - before) / b, 0.0, 1.0)))
    if not ratios:
        return None
    m = float(np.mean(ratios))
    n = len(ratios)
    w = n / (n + 1.0)
    return float(np.clip(w * m + (1.0 - w) * 0.5, 0.0, 1.0))

def annual_structure(history, total_length, period=PERIOD, half_window=3, slope_cap=0.3, horizon_damp=0.5, default_cred=0.75):
    y = np.asarray(history, float)
    L = len(y)
    eps = _eps(y)
    ly = np.log(np.maximum(y, 0.0) + eps)
    m = _backbone(ly, period)
    detr = ly - m
    ph_all = _phase(total_length, L, period)
    ph_past = ph_all[:L]
    if L >= period:
        prof = _profile(detr, ph_past, period, half_window)
        rel, ncyc = _reliability(detr, ph_past, prof, period, half_window)
    else:
        prof = np.zeros(period)
        rel, ncyc = (0.0, L / float(period))
    recent = ly[-min(L, 2 * period):]
    slope = _slope(recent, period, slope_cap)
    cred = _trend_credibility(ly, period, slope_cap)
    measured_cred = cred is not None
    if cred is None:
        cred = default_cred
    step = np.clip((np.arange(total_length) - (L - 1)) / float(period), 0.0, None)
    base = np.concatenate([m, np.full(max(total_length - L, 0), m[-1])])[:total_length]
    log_level = base + slope * cred * horizon_damp * step
    shape = np.sqrt(rel) * prof[ph_all]
    struct = np.maximum(np.exp(log_level + shape) - eps, 0.0)
    season = np.exp(shape)
    info = {'rel': float(rel), 'ncyc': float(ncyc), 'slope': float(slope), 'cred': float(cred), 'measured_cred': bool(measured_cred), 'amp': float(np.exp(shape).max() - np.exp(shape).min())}
    return {'struct': struct, 'season': season, 'shape_log': shape, 'log_level': log_level, 'eps': eps, 'info': info}

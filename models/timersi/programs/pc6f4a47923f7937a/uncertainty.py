import numpy as np
Z = np.array([-1.2815515655446004, -0.8416212335729143, -0.5244005127080407, -0.2533471031357997, 0.0, 0.2533471031357997, 0.5244005127080407, 0.8416212335729143, 1.2815515655446004])
_PAIRS = ((3, 5, 0.2533471031357997), (2, 6, 0.5244005127080407), (1, 7, 0.8416212335729143))

def _robust_sd(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    if x.size == 1:
        return abs(float(x[0]))
    mad = np.median(np.abs(x - np.median(x))) * 1.4826
    sd = float(np.std(x, ddof=1))
    if not np.isfinite(mad):
        mad = sd
    return 0.5 * (mad + sd)

def own_log_volatility(history, observed=None, cap=0.5):
    y = np.asarray(history, float)
    if observed is not None:
        y = y[np.asarray(observed, bool)]
    y = y[np.isfinite(y)]
    if y.size < 2:
        return (np.nan, 0)
    if np.all(y > 0):
        d = np.diff(np.log(y))
    else:
        lev = np.median(np.abs(y))
        lev = lev if lev > 0 else 1.0
        d = np.diff(y) / lev
    s = _robust_sd(d)
    if not np.isfinite(s) or s <= 0:
        s = np.nan
    return (min(s, cap) if np.isfinite(s) else np.nan, int(d.size))

def horizon_scale(sigma1, n_diffs, H):
    h = np.arange(1, H + 1, dtype=float)
    if not np.isfinite(sigma1) or sigma1 <= 0:
        return np.zeros(H)
    n = max(int(n_diffs), 1)
    return sigma1 * np.sqrt(h + h * h / n)

def model_log_scale(q, median):
    q = np.asarray(q, float)
    med = np.asarray(median, float)
    est = []
    for lo, hi, z in _PAIRS:
        a, b = (q[:, lo], q[:, hi])
        good = (a > 0) & (b > 0) & (med > 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            s = np.where(good, np.log(np.maximum(b, 1e-12) / np.maximum(a, 1e-12)) / (2 * z), np.nan)
        est.append(s)
    S = np.nanmedian(np.stack(est), axis=0)
    rel = np.abs(q[:, 7] - q[:, 1]) / (2 * 0.8416212335729143 * np.maximum(np.abs(med), 1e-12))
    S = np.where(np.isfinite(S) & (S > 0), S, rel)
    return np.maximum(S, 1e-06)

def recalibrate(q, median_path, sigma1, n_diffs, max_widen=4.0):
    q = np.asarray(q, float)
    H = q.shape[0]
    med0 = q[:, 4]
    sig_m = model_log_scale(q, med0)
    tgt = horizon_scale(sigma1, n_diffs, H)
    sig_f = np.sqrt(sig_m ** 2 + tgt ** 2)
    w = np.clip(sig_f / sig_m, 1.0, max_widen)
    newmed = np.asarray(median_path, float)
    out = np.empty_like(q)
    pos = (q > 0) & (med0[:, None] > 0) & (newmed[:, None] > 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        logdev = np.log(np.maximum(q, 1e-12)) - np.log(np.maximum(med0, 1e-12))[:, None]
        mult = newmed[:, None] * np.exp(w[:, None] * logdev)
    add = newmed[:, None] + w[:, None] * (q - med0[:, None])
    out = np.where(pos, mult, add)
    out = np.sort(out, axis=1)
    out[~np.isfinite(out)] = np.repeat(newmed[:, None], 9, axis=1)[~np.isfinite(out)]
    return out

def recalibrate_logspace(q_u, median_u, sigma1, n_diffs, max_widen=4.0):
    q_u = np.asarray(q_u, float)
    H = q_u.shape[0]
    med0 = q_u[:, 4]
    est = []
    for lo, hi, z in _PAIRS:
        est.append((q_u[:, hi] - q_u[:, lo]) / (2 * z))
    sig_m = np.maximum(np.nanmedian(np.stack(est), axis=0), 1e-06)
    tgt = horizon_scale(sigma1, n_diffs, H)
    w = np.clip(np.sqrt(sig_m ** 2 + tgt ** 2) / sig_m, 1.0, max_widen)
    newmed = np.asarray(median_u, float)
    out = np.exp(newmed[:, None] + w[:, None] * (q_u - med0[:, None]))
    return np.sort(out, axis=1)

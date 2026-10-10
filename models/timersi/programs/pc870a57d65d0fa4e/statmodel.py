import numpy as np
from scipy import stats
QUANTILE_LEVELS = np.arange(1, 10) / 10.0

def _weights(n, half_life):
    age = np.arange(n - 1, -1, -1, dtype=float)
    return np.exp(-np.log(2.0) * age / max(half_life, 1e-06))

def fit_drift(z, half_life=8.0, max_diffs=40):
    z = np.asarray(z, dtype=float).ravel()
    z = z[np.isfinite(z)]
    if z.size < 2:
        return {'level': float(z[-1]) if z.size else 0.0, 'drift': 0.0, 'sigma': 0.0, 'dof': 2.0, 'n': 0}
    d = np.diff(z)
    if d.size > max_diffs:
        d = d[-max_diffs:]
    n = d.size
    w = _weights(n, half_life)
    w = w / w.sum()
    drift = float(np.sum(w * d))
    if n >= 2:
        resid = d - drift
        var = float(np.sum(w * resid ** 2) / max(1.0 - np.sum(w ** 2), 1e-06))
        sigma = float(np.sqrt(max(var, 0.0)))
        mad = float(np.median(np.abs(d - np.median(d)))) * 1.4826
        sigma = float(np.sqrt(max(0.5 * sigma ** 2 + 0.5 * max(mad, 0.0) ** 2, 1e-12)))
    else:
        sigma = float(abs(d[0])) * 0.5
    eff = float(np.sum(w) ** 2 / max(np.sum(w ** 2), 1e-12))
    return {'level': float(z[-1]), 'drift': drift, 'sigma': sigma, 'dof': max(eff - 1.0, 2.0), 'n': n, 'eff': eff}

def drift_quantiles(fit, horizon, damp=1.0, sigma_mult=1.0, levels=QUANTILE_LEVELS):
    h = np.arange(1, horizon + 1, dtype=float)
    eff = max(fit.get('eff', max(fit['n'], 1.0)), 1.0)
    if damp >= 0.999:
        cum = h
    else:
        cum = np.cumsum(damp ** h)
    mu = fit['level'] + fit['drift'] * cum
    sigma = max(fit['sigma'], 1e-09) * sigma_mult
    sd = sigma * np.sqrt(h + h ** 2 / eff)
    tq = stats.t.ppf(levels, fit['dof'])
    return mu[:, None] + sd[:, None] * tq[None, :]

def naive_scale(z):
    z = np.asarray(z, dtype=float).ravel()
    if z.size < 2:
        return 1.0
    return float(np.mean(np.abs(np.diff(z)))) or 1.0

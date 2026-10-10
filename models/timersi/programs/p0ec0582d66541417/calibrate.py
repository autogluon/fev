import numpy as np
from calendar_tools import fourier_design, year_fraction
SHRINK_LOW = 0.8
SHRINK_HIGH = 0.88
VOL_EXPONENT = 0.5
VOL_CLIP = (0.75, 1.35)
LEVEL_SHIFT = 0.08
MIN_YEARS_FOR_VOL = 3.0

def mase_scale(history, observed=None):
    y = np.asarray(history, dtype=float)
    if y.size < 2:
        return np.array([1.0] * max(y.shape[0], 1))
    d = np.abs(np.diff(y, axis=-1))
    s = np.nanmean(d, axis=-1)
    med = np.nanmedian(np.abs(y), axis=-1)
    fallback = np.where(np.isfinite(med) & (med > 0), 0.01 * med, 1.0)
    return np.where(np.isfinite(s) & (s > 0), s, fallback)

def seasonal_volatility_ratio(history, past_stamps, future_stamps, spy, harmonics=4, max_years=8.0):
    y = np.asarray(history, dtype=float)
    n = y.shape[-1]
    if spy is None or n < MIN_YEARS_FOR_VOL * spy:
        return np.ones(y.shape[0])
    take = int(min(n, max_years * spy))
    ys = y[:, n - take:]
    ph = year_fraction(past_stamps[n - take:n])
    t = np.arange(take, dtype=float) / spy
    X = np.column_stack([fourier_design(ph, harmonics), t, t ** 2])
    bins = np.clip((ph * 52.0).astype(int), 0, 51)
    fut = np.clip((year_fraction(future_stamps) * 52.0).astype(int), 0, 51)
    out = np.ones(y.shape[0])
    for d in range(y.shape[0]):
        v = ys[d]
        pos = v[np.isfinite(v)]
        if pos.size < 10 or np.nanmin(v) <= 0:
            logv = v - np.nanmean(v)
        else:
            logv = np.log(v)
        good = np.isfinite(logv)
        if good.sum() < X.shape[1] + 5:
            continue
        beta, *_ = np.linalg.lstsq(X[good], logv[good], rcond=None)
        resid = np.abs(logv - X @ beta)
        prof = np.full(52, np.nan)
        for b in range(52):
            m = good & (bins == b)
            if m.any():
                prof[b] = resid[m].mean()
        overall = np.nanmean(prof)
        if not np.isfinite(overall) or overall <= 0:
            continue
        prof = np.where(np.isfinite(prof), prof, overall)
        smooth = np.array([prof[np.arange(b - 3, b + 4) % 52].mean() for b in range(52)])
        out[d] = float(np.clip(smooth[fut].mean() / overall, *VOL_CLIP))
    return out

def shrink(quantiles, vol_ratio, shrink_low=SHRINK_LOW, shrink_high=SHRINK_HIGH, exponent=VOL_EXPONENT):
    q = np.asarray(quantiles, dtype=float).copy()
    med = q[:, :, 4:5]
    dev = q - med
    m = np.asarray(vol_ratio, dtype=float).reshape(-1, 1, 1) ** exponent
    q = med + np.where(dev < 0.0, shrink_low * m * dev, shrink_high * m * dev)
    return np.sort(q, axis=-1)

def recentre(quantiles, scale, level_shift=LEVEL_SHIFT):
    q = np.asarray(quantiles, dtype=float) + (level_shift * np.asarray(scale, dtype=float)).reshape(-1, 1, 1)
    q = np.sort(q, axis=-1)
    return (q[:, :, 4].copy(), q)

def calibrate(point, quantiles, scale, vol_ratio):
    return recentre(shrink(quantiles, vol_ratio), scale)

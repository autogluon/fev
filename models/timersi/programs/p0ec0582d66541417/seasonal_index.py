import numpy as np
from calendar_tools import fourier_design, to_datetime64, year_fraction
HARMONICS = 3
MAX_YEARS = 10.0
MIN_YEARS = 3.0

def seasonal_shape(history, past_stamps, all_stamps, spy):
    y = np.asarray(history, dtype=float)
    D, n = y.shape
    out = np.ones((D, len(all_stamps)))
    if spy is None or n < MIN_YEARS * spy:
        return (out, False)
    take = int(min(n, MAX_YEARS * spy))
    ys = y[:, n - take:]
    ph = year_fraction(past_stamps[n - take:n])
    t = np.arange(take, dtype=float) / spy
    X = np.column_stack([fourier_design(ph, HARMONICS), t])
    ph_all = year_fraction(all_stamps)
    Xs = fourier_design(ph_all, HARMONICS)
    ok = False
    for d in range(D):
        v = ys[d]
        if not np.all(np.isfinite(v)) or np.min(v) <= 0:
            continue
        logv = np.log(v)
        beta, *_ = np.linalg.lstsq(X, logv, rcond=None)
        shape = Xs[:, 1:] @ beta[1:X.shape[1] - 1]
        shape = shape - shape.mean()
        shape = np.clip(shape, -1.5, 1.5)
        out[d] = np.exp(shape)
        ok = True
    return (out, ok)
YEAR_SECONDS = 365.2425 * 86400.0

def calendar_aligned_lag(history, all_stamps, cutoff, years):
    y = np.asarray(history, dtype=float)
    t = to_datetime64(all_stamps).astype('datetime64[s]').astype('float64')
    tp = t[:cutoff]
    if cutoff < 2 or tp[-1] - tp[0] < years * YEAR_SECONDS * 1.05:
        return None
    want = np.clip(t - years * YEAR_SECONDS, tp[0], tp[-1])
    return np.vstack([np.interp(want, tp, y[d]) for d in range(y.shape[0])])

def build_known_future(history, all_stamps, cutoff, spy):
    ph = year_fraction(all_stamps)
    chans = [np.sin(2.0 * np.pi * ph), np.cos(2.0 * np.pi * ph)]
    names = ['cal_sin_annual', 'cal_cos_annual']
    shape, ok = seasonal_shape(history, all_stamps[:cutoff], all_stamps, spy)
    if ok:
        for d in range(shape.shape[0]):
            chans.append(shape[d])
            names.append('own_seasonal_shape_%d' % d)
    for years in (1, 2):
        lag = calendar_aligned_lag(history, all_stamps, cutoff, years)
        if lag is None:
            continue
        for d in range(lag.shape[0]):
            chans.append(lag[d])
            names.append('cal_lag%dy_%d' % (years, d))
    return (np.asarray(chans, dtype=float), names)

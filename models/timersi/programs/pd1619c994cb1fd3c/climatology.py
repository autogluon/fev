import numpy as np
PEAK_DOY = 200.0
YEAR = 365.25
TEMPERATURE_TOKENS = ('OT', 'TEMP', 'TEMPERATURE', 'OIL')

def day_of_year(timestamps):
    ts = np.asarray(timestamps)
    if ts.dtype.kind in 'US':
        ts = ts.astype('datetime64[s]')
    day = ts.astype('datetime64[D]')
    return (day - day.astype('datetime64[Y]')).astype(int) + 1.0

def climate_basis(timestamps, peak=PEAK_DOY):
    return np.cos(2.0 * np.pi * (day_of_year(timestamps) - peak) / YEAR)

def is_temperature_target(name):
    token = str(name).upper()
    return any((token == t or token.endswith(t) or t in token.split('_') for t in TEMPERATURE_TOKENS))

def identifiability(s_past):
    if len(s_past) < 4:
        return 0.0
    return float(np.clip(0.5 * (np.nanmax(s_past) - np.nanmin(s_past)), 0.0, 1.0))

def fit_seasonal(y, s_past, min_t=1.5, shrink_exponent=0.55, max_amp_ratio=1.5):
    y = np.asarray(y, dtype=float)
    s = np.asarray(s_past, dtype=float)
    n = len(y)
    diag = {'n': int(n), 'raw_slope': 0.0, 'se': 0.0, 't': 0.0, 'kappa': 0.0, 'r': 0.0}
    if n < 6:
        return (0.0, 0.0, diag)
    sc = s - s.mean()
    denom = float((sc * sc).sum())
    if denom <= 1e-09:
        return (0.0, 0.0, diag)
    b = float((sc * (y - y.mean())).sum() / denom)
    resid = y - (y.mean() + b * sc)
    dof = max(n - 2, 1)
    sd = float(np.sqrt((resid ** 2).sum() / dof))
    se = sd / np.sqrt(denom) if denom > 0 else np.inf
    tstat = b / se if se > 0 else 0.0
    r = identifiability(s)
    kappa = r ** shrink_exponent
    if abs(tstat) < min_t:
        kappa = 0.0
    slope = kappa * b
    span = float(np.nanmax(y) - np.nanmin(y))
    if span > 0 and abs(slope) * 2.0 > max_amp_ratio * span / max(r, 0.001):
        slope = np.sign(slope) * max_amp_ratio * span / (2.0 * max(r, 0.001))
    diag.update({'raw_slope': b, 'se': float(se), 't': float(tstat), 'kappa': float(kappa), 'r': float(r), 'slope': float(slope)})
    return (float(slope), float(kappa), diag)

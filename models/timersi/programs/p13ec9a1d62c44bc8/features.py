import datetime as dt
import numpy as np

def _nth_weekday(year, month, weekday, n):
    d = dt.date(year, month, 1)
    d += dt.timedelta(days=(weekday - d.weekday()) % 7 + 7 * (n - 1))
    return d

def _last_weekday(year, month, weekday):
    d = dt.date(year, month, 28)
    while d.month == month:
        d += dt.timedelta(days=1)
    d -= dt.timedelta(days=1)
    return d - dt.timedelta(days=(d.weekday() - weekday) % 7)

def us_holidays(years):
    out = set()
    for y in years:
        out |= {dt.date(y, 1, 1), dt.date(y, 7, 4), dt.date(y, 11, 11), dt.date(y, 12, 25), _nth_weekday(y, 1, 0, 3), _nth_weekday(y, 2, 0, 3), _last_weekday(y, 5, 0), _nth_weekday(y, 9, 0, 1), _nth_weekday(y, 10, 0, 2), _nth_weekday(y, 11, 3, 4)}
    return {np.datetime64(d) for d in out}

def calendar_parts(ts):
    days = ts.astype('datetime64[D]')
    hour = (ts - days).astype('timedelta64[h]').astype(int)
    epoch_day = days.astype('datetime64[D]').astype(int)
    dow = (epoch_day + 4) % 7
    yrs = np.unique(ts.astype('datetime64[Y]').astype(int) + 1970)
    hol = us_holidays(range(int(yrs.min()) - 1, int(yrs.max()) + 2))
    is_hol = np.fromiter((d in hol for d in days), float, len(days))
    doy = (days - ts.astype('datetime64[Y]').astype('datetime64[D]')).astype(int)
    return (hour, dow, is_hol, doy, epoch_day)

def ewma_causal(x, halflife):
    a = 1.0 - 0.5 ** (1.0 / halflife)
    out = np.empty_like(x, dtype=float)
    acc = float(x[0])
    for i in range(len(x)):
        acc += a * (float(x[i]) - acc)
        out[i] = acc
    return out
FEATURE_NAMES = None

def build_matrix(ts, temp):
    global FEATURE_NAMES
    hour, dow, is_hol, doy, epoch_day = calendar_parts(ts)
    T = np.asarray(temp, float)
    s12, s48, s168 = (ewma_causal(T, 12), ewma_causal(T, 48), ewma_causal(T, 168))
    nonwork = np.maximum((dow >= 5).astype(float), is_hol)
    shape = np.sin(2 * np.pi * (hour - 9) / 24.0)
    feats = {'T': T, 'T2': T * T, 'cdd65': np.maximum(T - 65.0, 0.0), 'hdd60': np.maximum(60.0 - T, 0.0), 'cdd75': np.maximum(T - 75.0, 0.0), 'hdd45': np.maximum(45.0 - T, 0.0), 'Ts12': s12, 'Ts48': s48, 'Ts168': s168, 'cdd_s48': np.maximum(s48 - 65.0, 0.0), 'hdd_s48': np.maximum(60.0 - s48, 0.0), 'hour': hour.astype(float), 'dow': dow.astype(float), 'nonwork': nonwork, 'is_hol': is_hol, 'hsin': np.sin(2 * np.pi * hour / 24.0), 'hcos': np.cos(2 * np.pi * hour / 24.0), 'hsin2': np.sin(4 * np.pi * hour / 24.0), 'hcos2': np.cos(4 * np.pi * hour / 24.0), 'doysin': np.sin(2 * np.pi * doy / 365.25), 'doycos': np.cos(2 * np.pi * doy / 365.25), 'trend': (epoch_day - epoch_day[0]).astype(float), 'cdd_shape': np.maximum(T - 65.0, 0.0) * shape, 'hdd_shape': np.maximum(60.0 - T, 0.0) * shape}
    FEATURE_NAMES = sorted(feats)
    X = np.column_stack([feats[n] for n in FEATURE_NAMES])
    return (np.ascontiguousarray(X, dtype=np.float64), FEATURE_NAMES)

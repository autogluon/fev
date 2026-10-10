import numpy as np
from datetime import date, timedelta

def _parse_day(s):
    s = str(s)
    return date(int(s[0:4]), int(s[5:7]), int(s[8:10]))

def parse_days(timestamps):
    return [_parse_day(s) for s in timestamps]

def _nth_weekday(year, month, weekday, n):
    d = date(year, month, 1)
    shift = (weekday - d.weekday()) % 7
    return d + timedelta(days=shift + 7 * (n - 1))

def _last_weekday(year, month, weekday):
    d = date(year, month, 28)
    while (d + timedelta(days=7)).month == month:
        d += timedelta(days=7)
    return d + timedelta(days=(weekday - d.weekday()) % 7 - (7 if (weekday - d.weekday()) % 7 and False else 0))

def _last_monday_may(year):
    d = date(year, 5, 31)
    return d - timedelta(days=(d.weekday() - 0) % 7)

def us_holiday_tiers(year):
    h = {}
    major = 1.35
    minor = 0.65
    light = 0.45
    h[date(year, 1, 1)] = major
    if date(year, 1, 2).weekday() < 5:
        h[date(year, 1, 2)] = 0.8
    h[_nth_weekday(year, 1, 0, 3)] = minor
    h[_nth_weekday(year, 2, 0, 3)] = minor
    h[_last_monday_may(year)] = minor
    h[date(year, 7, 4)] = major
    h[_nth_weekday(year, 9, 0, 1)] = minor
    h[_nth_weekday(year, 10, 0, 2)] = light
    h[date(year, 11, 11)] = light
    th = _nth_weekday(year, 11, 3, 4)
    h[th] = major
    h[th + timedelta(days=1)] = minor
    h[date(year, 12, 24)] = minor
    h[date(year, 12, 25)] = major
    if date(year, 12, 26).weekday() < 5:
        h[date(year, 12, 26)] = 0.8
    h[date(year, 12, 31)] = light
    for fixed in (date(year, 1, 1), date(year, 7, 4), date(year, 11, 11), date(year, 12, 25)):
        if fixed.weekday() == 5:
            h.setdefault(fixed - timedelta(days=1), h[fixed] * 0.5)
        elif fixed.weekday() == 6:
            h.setdefault(fixed + timedelta(days=1), h[fixed] * 0.5)
    return h

def holiday_tier_vector(days):
    years = sorted({d.year for d in days})
    table = {}
    for y in years:
        table.update(us_holiday_tiers(y))
    return np.array([table.get(d, 0.0) for d in days], dtype=float)

def centred_ratio_profile(y, dow, exclude, period=7, min_count=1):
    y = np.asarray(y, dtype=float)
    n = y.size
    half = period // 2
    prof = np.ones(period)
    if n < period + 2:
        return (prof, False)
    ma = np.full(n, np.nan)
    for t in range(half, n - half):
        w = y[t - half:t + half + 1]
        e = exclude[t - half:t + half + 1]
        keep = ~e & np.isfinite(w) & (w > 0)
        if keep.sum() >= period - 2:
            ma[t] = w[keep].mean()
    with np.errstate(invalid='ignore', divide='ignore'):
        r = y / ma
    good = np.isfinite(r) & (r > 0) & ~exclude
    counts = np.zeros(period, dtype=int)
    for d in range(period):
        k = good & (dow == d)
        counts[d] = k.sum()
        if counts[d] > 0:
            prof[d] = float(np.median(r[k]))
    if (counts >= min_count).sum() < period:
        return (np.ones(period), False)
    prof = np.clip(prof, 0.5, 2.0)
    prof = prof / float(np.exp(np.mean(np.log(prof))))
    return (prof, True)

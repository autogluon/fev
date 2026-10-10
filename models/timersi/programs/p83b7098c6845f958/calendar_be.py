import numpy as np
BE_FIXED = {(1, 1), (5, 1), (7, 21), (8, 15), (11, 1), (11, 11), (12, 25), (12, 26)}

def _easter(year):
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = (h + l - 7 * m + 114) % 31 + 1
    return np.datetime64('%04d-%02d-%02d' % (year, month, day))

def time_axis(timestamps):
    ts = np.asarray([str(t)[:19] for t in timestamps], dtype='datetime64[s]').astype('datetime64[h]')
    hour = (ts.astype('datetime64[h]').astype(np.int64) % 24).astype(np.int64)
    days = ts.astype('datetime64[D]')
    dow = ((days.astype(np.int64) + 3) % 7).astype(np.int64)
    return (ts, hour, dow, days)

def offday_flag(timestamps):
    ts, hour, dow, days = time_axis(timestamps)
    years = days.astype('datetime64[Y]').astype(int) + 1970
    months = days.astype('datetime64[M]').astype(int) % 12 + 1
    doms = (days - days.astype('datetime64[M]')).astype(int) + 1
    flag = (dow >= 5).astype(float)
    for mm, dd in BE_FIXED:
        flag[(months == mm) & (doms == dd)] = 1.0
    mov = []
    for y in np.unique(years):
        e = _easter(int(y))
        for off in (1, 39, 50):
            mov.append(e + np.timedelta64(off, 'D'))
    if mov:
        flag[np.isin(days, np.array(mov, dtype='datetime64[D]'))] = 1.0
    return flag

def daytype_hour_profile(target, origin, hour, offday, weeks=8):
    y = np.asarray(target, float)
    lo = max(0, int(origin) - 24 * 7 * int(weeks))
    hh, dd, yy = (hour[lo:origin], (offday[lo:origin] > 0.5).astype(int), y[lo:origin])
    good = np.isfinite(yy)
    overall = float(np.median(yy[good])) if good.any() else 0.0
    prof = np.full((2, 24), overall)
    for t in range(2):
        for h in range(24):
            m = good & (hh == h) & (dd == t)
            if m.sum() >= 3:
                prof[t, h] = float(np.median(yy[m]))
    return prof[(offday > 0.5).astype(int), hour]

def trailing_same_hour_ratio(series, origin, window_days=28, eps=1e-06):
    x = np.asarray(series, float)
    n = x.shape[0]
    out = np.ones(n)
    w = int(window_days)
    for p in range(24):
        v = x[p::24]
        k = np.arange(v.size)
        cs = np.concatenate([[0.0], np.cumsum(v)])
        lo = np.maximum(k - w, 0)
        cnt = k - lo
        tot = cs[k] - cs[lo]
        mean = np.where(cnt >= 4, tot / np.maximum(cnt, 1), np.nan)
        oi = max(0, (int(origin) - p + 23) // 24)
        if oi < mean.size:
            last = mean[min(oi, mean.size - 1)]
            if np.isfinite(last):
                mean[oi:] = last
        r = np.where(np.isfinite(mean) & (np.abs(mean) > eps), v / (mean + eps), 1.0)
        out[p::24] = np.clip(r, 0.2, 5.0)
    return out

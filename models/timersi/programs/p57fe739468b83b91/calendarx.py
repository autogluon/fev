import numpy as np
_FIXED = ((1, 1), (7, 4), (11, 11), (12, 24), (12, 25), (12, 31))

def _nth_weekday(year, month, weekday, n):
    first = np.datetime64('%04d-%02d-01' % (year, month), 'D')
    if n > 0:
        w0 = (first.astype(int) + 3) % 7
        day = 1 + (weekday - w0) % 7 + 7 * (n - 1)
        return day
    nxt = first + np.timedelta64(32, 'D')
    nxt = np.datetime64('%04d-%02d-01' % (int(str(nxt)[:4]), int(str(nxt)[5:7])), 'D')
    last = nxt - np.timedelta64(1, 'D')
    wl = (last.astype(int) + 3) % 7
    return int(str(last)[8:10]) - (wl - weekday) % 7

def us_holidays(years):
    out = set()
    for y in years:
        for m, d in _FIXED:
            out.add(np.datetime64('%04d-%02d-%02d' % (y, m, d), 'D'))
        out.add(np.datetime64('%04d-01-%02d' % (y, _nth_weekday(y, 1, 0, 3)), 'D'))
        out.add(np.datetime64('%04d-02-%02d' % (y, _nth_weekday(y, 2, 0, 3)), 'D'))
        out.add(np.datetime64('%04d-05-%02d' % (y, _nth_weekday(y, 5, 0, -1)), 'D'))
        out.add(np.datetime64('%04d-09-%02d' % (y, _nth_weekday(y, 9, 0, 1)), 'D'))
        out.add(np.datetime64('%04d-10-%02d' % (y, _nth_weekday(y, 10, 0, 2)), 'D'))
        out.add(np.datetime64('%04d-11-%02d' % (y, _nth_weekday(y, 11, 3, 4)), 'D'))
        out.add(np.datetime64('%04d-11-%02d' % (y, _nth_weekday(y, 11, 3, 4) + 1), 'D'))
    extra = set()
    for d in out:
        w = (d.astype(int) + 3) % 7
        if w == 5:
            extra.add(d - np.timedelta64(1, 'D'))
        elif w == 6:
            extra.add(d + np.timedelta64(1, 'D'))
    return out | extra

def day_type(ts):
    d = ts.astype('datetime64[D]')
    w = (d.astype('int64') + 3) % 7
    years = range(int(str(d.min())[:4]), int(str(d.max())[:4]) + 1)
    hol = us_holidays(years)
    ish = np.isin(d, np.array(sorted(hol), dtype='datetime64[D]'))
    v = np.where(w == 6, 1.0, np.where((w == 5) | ish, 0.5, 0.0))
    return v

def temp_channels(T, base=65.0, scale=25.0):
    T = np.asarray(T, dtype=float)
    n = T.size
    cs = np.concatenate([[0.0], np.cumsum(T)])
    idx = np.arange(n)
    lo = np.maximum(idx - 23, 0)
    ma24 = (cs[idx + 1] - cs[lo]) / (idx - lo + 1)
    hdd = np.maximum(base - T, 0.0) / scale
    cdd = np.maximum(T - base, 0.0) / scale
    hdd24 = np.maximum(base - ma24, 0.0) / scale
    return (ma24, hdd, cdd, hdd24)

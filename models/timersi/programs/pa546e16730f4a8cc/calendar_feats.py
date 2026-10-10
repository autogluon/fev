import datetime as _dt
import numpy as np

def us_holidays(years):
    out = set()

    def nth_wd(y, m, wd, n):
        d = _dt.date(y, m, 1)
        d += _dt.timedelta((wd - d.weekday()) % 7)
        return d + _dt.timedelta(7 * (n - 1))
    for y in years:
        out |= {_dt.date(y, 1, 1), _dt.date(y, 7, 4), _dt.date(y, 12, 25), _dt.date(y, 12, 24), _dt.date(y, 12, 31), _dt.date(y, 11, 11)}
        out |= {nth_wd(y, 1, 0, 3), nth_wd(y, 2, 0, 3), nth_wd(y, 9, 0, 1), nth_wd(y, 10, 0, 2)}
        d = _dt.date(y, 5, 31)
        while d.weekday() != 0:
            d -= _dt.timedelta(1)
        out.add(d)
        tg = nth_wd(y, 11, 3, 4)
        out |= {tg, tg + _dt.timedelta(1)}
    return out

def parse_times(timestamps):
    hours = np.empty(len(timestamps), dtype=np.int64)
    wdays = np.empty(len(timestamps), dtype=np.int64)
    dates = []
    for i, t in enumerate(timestamps):
        s = str(t)[:19]
        d = _dt.datetime.strptime(s, '%Y-%m-%dT%H:%M:%S') if 'T' in s else _dt.datetime.strptime(s, '%Y-%m-%d %H:%M:%S')
        hours[i] = d.hour
        wdays[i] = d.weekday()
        dates.append(d.date())
    return (hours, wdays, dates)

def offday_flag(wdays, dates):
    years = sorted({d.year for d in dates})
    hol = us_holidays(range(min(years) - 1, max(years) + 2))
    flag = np.array([1.0 if wdays[i] >= 5 or dates[i] in hol else 0.0 for i in range(len(dates))])
    return (flag, hol)

def trailing_hourly_norm(series, cutoff, window_days=28, eps=1e-06):
    x = np.asarray(series, float)
    n = len(x)
    out = np.ones(n)
    w = int(window_days)
    for p in range(24):
        v = x[p::24]
        if len(v) < 5:
            continue
        cs = np.concatenate([[0.0], np.cumsum(v)])
        k = np.arange(len(v))
        lo = np.maximum(k - w, 0)
        cnt = k - lo
        tot = cs[k] - cs[lo]
        m = np.where(cnt >= 4, tot / np.maximum(cnt, 1), np.nan)
        r = np.where(np.isfinite(m) & (np.abs(m) > eps), v / (m + eps), 1.0)
        out[p::24] = r
    return out

def causal_calendar_profile(target, cutoff, hours, wdays, offday, weeks=8):
    y = np.asarray(target, float)
    n = len(hours)
    look = min(cutoff, 24 * 7 * weeks)
    lo = cutoff - look
    hh = hours[lo:cutoff]
    dd = (offday[lo:cutoff] > 0.5).astype(int)
    yy = y[lo:cutoff]
    good = np.isfinite(yy)
    prof = np.zeros((2, 24))
    overall = np.median(yy[good]) if good.any() else 0.0
    for t in range(2):
        for h in range(24):
            m = good & (hh == h) & (dd == t)
            prof[t, h] = np.median(yy[m]) if m.sum() >= 3 else overall
    out = np.empty(n)
    dt_all = (offday > 0.5).astype(int)
    for i in range(n):
        out[i] = prof[dt_all[i], hours[i]]
    return (out, prof)

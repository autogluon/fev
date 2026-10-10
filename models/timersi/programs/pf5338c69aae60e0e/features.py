import numpy as np
DAY = np.timedelta64(1, 'D')

def to_days(timestamps):
    return np.array(timestamps, dtype='datetime64[D]')

def dayofweek(d):
    return (d.astype('int64') + 3) % 7

def year_of(d):
    return d.astype('datetime64[Y]').astype(int) + 1970

def doy_frac(d):
    y0 = d.astype('datetime64[Y]').astype('datetime64[D]')
    y1 = (d.astype('datetime64[Y]') + 1).astype('datetime64[D]')
    return (d - y0).astype(float) / (y1 - y0).astype(float)

def nth_weekday(year, month, weekday, n):
    first = np.datetime64(f'{year:04d}-{month:02d}-01', 'D')
    off = (weekday - int(dayofweek(first))) % 7
    return first + np.timedelta64(off + 7 * (n - 1), 'D')

def last_weekday(year, month, weekday):
    nxt = np.datetime64(f'{year:04d}-{month:02d}-01', 'D') + np.timedelta64(32, 'D')
    last = nxt.astype('datetime64[M]').astype('datetime64[D]') - DAY
    return last - np.timedelta64(int((int(dayofweek(last)) - weekday) % 7), 'D')

def holiday_map(years):
    out = {k: [] for k in ('newyear', 'memorial', 'july4', 'labor', 'thanksgiving', 'blackfriday', 'thxwknd', 'xmas_eve', 'xmas', 'xmas_wk', 'newyear_eve')}
    for y in years:
        out['newyear'].append(np.datetime64(f'{y}-01-01', 'D'))
        out['memorial'].append(last_weekday(y, 5, 0))
        out['july4'].append(np.datetime64(f'{y}-07-04', 'D'))
        out['labor'].append(nth_weekday(y, 9, 0, 1))
        th = nth_weekday(y, 11, 3, 4)
        out['thanksgiving'].append(th)
        out['blackfriday'].append(th + DAY)
        out['thxwknd'] += [th + 2 * DAY, th + 3 * DAY]
        out['xmas_eve'].append(np.datetime64(f'{y}-12-24', 'D'))
        out['xmas'].append(np.datetime64(f'{y}-12-25', 'D'))
        out['xmas_wk'] += [np.datetime64(f'{y}-12-26', 'D') + i * DAY for i in range(4)]
        out['newyear_eve'].append(np.datetime64(f'{y}-12-31', 'D'))
    return {k: np.array(sorted(set(v))) for k, v in out.items()}

def holiday_indicators(dates):
    years = range(int(year_of(dates).min()) - 1, int(year_of(dates).max()) + 2)
    hmap = holiday_map(years)
    keys = sorted(hmap)
    M = np.zeros((len(keys), len(dates)))
    for r, k in enumerate(keys):
        M[r] = np.isin(dates, hmap[k]).astype(float)
    return (keys, M)

def centered_median(x, win):
    n = len(x)
    half = win // 2
    pad = np.concatenate([x[:half][::-1], x, x[-half:][::-1]])
    out = np.empty(n)
    from numpy.lib.stride_tricks import sliding_window_view
    w = sliding_window_view(pad, 2 * half + 1)
    out[:] = np.median(w[:n], axis=1)
    return out

class CalendarModel:

    def __init__(self, y, dates, dow_years=3.0, hol_years=12.0, n_harm=4):
        self.ok = False
        y = np.asarray(y, float)
        pos = y[np.isfinite(y)]
        if len(pos) < 400 or np.nanmin(y) <= 0:
            self.shift = 0.0
            return
        self.dates = dates
        ly = np.log(y)
        base = centered_median(ly, 29)
        resid = ly - base
        d = dayofweek(dates)
        n = len(ly)
        recent = np.arange(n) >= n - int(365.25 * dow_years)
        dowf = np.zeros(7)
        for j in range(7):
            m = recent & (d == j)
            if m.sum() >= 8:
                dowf[j] = np.median(resid[m])
        dowf -= dowf.mean()
        self.dowf = dowf
        r2 = resid - dowf[d]
        keys, M = holiday_indicators(dates)
        hw = np.arange(n) >= n - int(365.25 * hol_years)
        coef = {}
        for r, k in enumerate(keys):
            m = (M[r] > 0) & hw
            if m.sum() >= 4:
                coef[k] = float(np.median(r2[m]))
            else:
                coef[k] = 0.0
        self.hol_coef = coef
        self.hol_keys = keys
        trend = centered_median(ly, 365)
        anom = ly - trend - dowf[d]
        ff = doy_frac(dates)
        cols = [np.ones(n)]
        for h in range(1, n_harm + 1):
            cols += [np.sin(2 * np.pi * h * ff), np.cos(2 * np.pi * h * ff)]
        X = np.stack(cols, 1)
        use = np.arange(n) >= max(0, n - int(365.25 * 10))
        beta, *_ = np.linalg.lstsq(X[use], anom[use], rcond=None)
        self.beta = beta
        self.n_harm = n_harm
        self.ok = True

    def dow_feature(self, dates):
        return self.dowf[dayofweek(dates)] if self.ok else np.zeros(len(dates))

    def holiday_feature(self, dates):
        if not self.ok:
            return np.zeros(len(dates))
        keys, M = holiday_indicators(dates)
        out = np.zeros(len(dates))
        for r, k in enumerate(keys):
            out += self.hol_coef.get(k, 0.0) * M[r]
        return out

    def annual_feature(self, dates):
        if not self.ok:
            return np.zeros(len(dates))
        ff = doy_frac(dates)
        cols = [np.ones(len(dates))]
        for h in range(1, self.n_harm + 1):
            cols += [np.sin(2 * np.pi * h * ff), np.cos(2 * np.pi * h * ff)]
        v = np.stack(cols, 1) @ self.beta
        return v - v.mean()

def climatology_level(y, dates, fut_dates, n_years=10, win=15, ref_win=28, trend_years=3):
    y = np.asarray(y, float)
    ly = np.log(np.maximum(y, 1e-06))
    n = len(ly)
    H = len(fut_dates)
    ref = ly[-ref_win:].mean()
    deltas = np.full((n_years, H), np.nan)
    for k in range(1, n_years + 1):
        oi = n - 1 - int(round(365.25 * k))
        if oi - ref_win < 0:
            continue
        base = ly[oi - ref_win + 1:oi + 1].mean()
        for h in range(H):
            ti, lo, hi = (oi + 1 + h, oi + 1 + h - win, oi + 2 + h + win)
            if lo < 0 or hi > n:
                continue
            deltas[k - 1, h] = ly[lo:hi].mean() - base
    with np.errstate(invalid='ignore', all='ignore'):
        delta = np.nanmedian(deltas, axis=0)
    delta = np.where(np.isfinite(delta), delta, 0.0)
    growth = 0.0
    if n > 365 * (trend_years + 1):
        growth = (ly[-365:].mean() - ly[-365 * (trend_years + 1):-365 * trend_years].mean()) / trend_years
        growth = float(np.clip(growth, -0.1, 0.1))
    hy = (np.arange(1, H + 1) + ref_win / 2.0) / 365.25
    return ref + delta + growth * hy

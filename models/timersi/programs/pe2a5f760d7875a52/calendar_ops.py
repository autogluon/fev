import calendar
import datetime as _dt
import numpy as np
_HARM = (1, 2)
N_PAYDAY = 2 * len(_HARM)
N_FEATURES = N_PAYDAY + 2
RIDGE = np.array([4.0] * N_PAYDAY + [3.0, 3.0])

def parse_day(ts):
    s = str(ts)[:10]
    return _dt.date(int(s[:4]), int(s[5:7]), int(s[8:10]))

def easter(year):
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
    return _dt.date(year, month, day)

def mothers_day(year):
    d = _dt.date(year, 5, 1)
    d += _dt.timedelta(days=(6 - d.weekday()) % 7)
    return d + _dt.timedelta(days=7)

def _event_days(year):
    mov, fix = ({}, {})
    e = easter(year)
    for off, w in ((-50, 0.7), (-49, 0.7), (-48, 1.0), (-47, 1.0)):
        mov[e + _dt.timedelta(days=off)] = w
    for off in (-3, -2, -1, 0):
        mov[e + _dt.timedelta(days=off)] = max(mov.get(e + _dt.timedelta(days=off), 0.0), 0.8)
    md = mothers_day(year)
    for off in (-2, -1, 0):
        mov[md + _dt.timedelta(days=off)] = 0.8
    for day, w in ((20, 0.6), (21, 0.6), (22, 0.6), (23, 1.0), (24, 1.0), (25, 1.0), (26, 0.6), (27, 0.6), (28, 0.6), (29, 0.6), (30, 1.0), (31, 1.0)):
        fix[_dt.date(year, 12, day)] = w
    fix[_dt.date(year, 1, 1)] = 1.0
    fix[_dt.date(year, 1, 2)] = 0.6
    return (mov, fix)
_CACHE = {}

def _year(y):
    if y not in _CACHE:
        _CACHE[y] = _event_days(y)
    return _CACHE[y]

def week_features(timestamps):
    rows = []
    for ts in timestamps:
        end = parse_day(ts)
        frac, mov, fix = ([], 0.0, 0.0)
        for back in range(7):
            d = end - _dt.timedelta(days=back)
            nd = calendar.monthrange(d.year, d.month)[1]
            frac.append((d.day - 1) / float(max(nd - 1, 1)))
            m, f = _year(d.year)
            mov += m.get(d, 0.0)
            fix += f.get(d, 0.0)
        frac = np.asarray(frac)
        row = []
        for k in _HARM:
            row.append(float(np.cos(2 * np.pi * k * frac).mean()))
            row.append(float(np.sin(2 * np.pi * k * frac).mean()))
        row += [mov / 7.0, fix / 7.0]
        rows.append(row)
    return np.asarray(rows, float)

def rolling_centered_median(x, win=13):
    n = len(x)
    out = np.full(n, np.nan)
    half = win // 2
    need = max(7, half)
    for t in range(n):
        seg = x[max(0, t - half):min(n, t + half + 1)]
        seg = seg[np.isfinite(seg)]
        if len(seg) >= need:
            out[t] = np.median(seg)
    return out

def fit_calendar(y, X_past, min_weeks=26, warm=20.0, full=32.0, win=13):
    y = np.asarray(y, float)
    pos = np.isfinite(y) & (y > 0)
    if len(y) < min_weeks or pos.sum() < 24:
        return (np.zeros(N_FEATURES), 0.0)
    lg = np.where(pos, np.log(np.maximum(y, 1e-09)), np.nan)
    resid = lg - rolling_centered_median(lg, win)
    ok = np.isfinite(resid)
    Xo, r = (X_past[ok], resid[ok])
    if len(r) < 24:
        return (np.zeros(N_FEATURES), 0.0)
    beta = np.zeros(N_FEATURES)
    keep = np.asarray([Xo[:, j].std() > 1e-08 for j in range(N_FEATURES)])
    if not keep.any():
        return (beta, 0.0)
    try:
        beta[keep] = np.linalg.solve(Xo[:, keep].T @ Xo[:, keep] + np.diag(RIDGE[keep]), Xo[:, keep].T @ r)
    except np.linalg.LinAlgError:
        return (np.zeros(N_FEATURES), 0.0)
    if not np.all(np.isfinite(beta)):
        return (np.zeros(N_FEATURES), 0.0)
    return (beta, float(min(1.0, max(0.0, (len(r) - warm) / full))))

def applied_design(X_future):
    Xa = X_future.copy()
    Xa[:, -1] = 0.0
    return Xa

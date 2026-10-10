import numpy as np
_SEC_DAY = 86400

def _last_sunday_utc(year, month):
    days = [31, 29 if year % 4 == 0 and (year % 100 or year % 400 == 0) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    d = days[month - 1]
    while True:
        t = _ymd_to_unix(year, month, d)
        if (t // _SEC_DAY + 4) % 7 == 0:
            return t + 3600
        d -= 1

def _ymd_to_unix(y, m, d):
    a = (14 - m) // 12
    yy = y + 4800 - a
    mm = m + 12 * a - 3
    jdn = d + (153 * mm + 2) // 5 + 365 * yy + yy // 4 - yy // 100 + yy // 400 - 32045
    return (jdn - 2440588) * _SEC_DAY

def utc_offset_seconds(unix_sec):
    u = np.asarray(unix_sec, dtype=np.int64)
    off = np.full(u.shape, 3600, dtype=np.int64)
    for year in range(int(_year_of(u.min())) - 1, int(_year_of(u.max())) + 2):
        start = _last_sunday_utc(year, 3)
        end = _last_sunday_utc(year, 10)
        off[(u >= start) & (u < end)] = 7200
    return off

def _year_of(unix_sec):
    return 1970 + int(unix_sec) // 31556952

def civil_fields(unix_sec):
    u = np.asarray(unix_sec, dtype=np.int64) + utc_offset_seconds(unix_sec)
    days = u // _SEC_DAY
    secs = u - days * _SEC_DAY
    tod = (secs // 1800).astype(np.int64)
    dow = ((days + 3) % 7).astype(np.int64)
    y, m, d = _civil_from_days(days)
    return (y, m, d, tod, dow, days)

def _civil_from_days(z):
    z = np.asarray(z, dtype=np.int64) + 719468
    era = np.where(z >= 0, z, z - 146096) // 146097
    doe = z - era * 146097
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    y = yoe + era * 400
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    mp = (5 * doy + 2) // 153
    d = doy - (153 * mp + 2) // 5 + 1
    m = mp + np.where(mp < 10, 3, -9)
    return (y + (m <= 2), m, d)

def _easter_days(year):
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = (h + l - 7 * m + 114) % 31 + 1
    return _ymd_to_unix(year, month, day) // _SEC_DAY
_FIXED = {'DE': [(1, 1), (5, 1), (10, 3), (12, 25), (12, 26)], 'AT': [(1, 1), (1, 6), (5, 1), (8, 15), (10, 26), (11, 1), (12, 8), (12, 25), (12, 26)], 'BE': [(1, 1), (5, 1), (7, 21), (8, 15), (11, 1), (11, 11), (12, 25)], 'LU': [(1, 1), (5, 1), (5, 9), (6, 23), (8, 15), (11, 1), (12, 25), (12, 26)], 'NL': [(1, 1), (4, 27), (12, 25), (12, 26)], 'HU': [(1, 1), (3, 15), (5, 1), (8, 20), (10, 23), (11, 1), (12, 25), (12, 26)]}
_EASTER = {'DE': [-2, 1, 39, 50], 'AT': [1, 39, 50, 60], 'BE': [1, 39, 50], 'LU': [1, 39, 50], 'NL': [-2, 0, 1, 39, 49, 50], 'HU': [-2, 1, 50]}
_DEFAULT = 'DE'

def holiday_mask(item_id, y, m, d, days):
    code = (item_id or _DEFAULT).split('_')[0].upper()[:2]
    fixed = _FIXED.get(code, _FIXED[_DEFAULT])
    easter = _EASTER.get(code, _EASTER[_DEFAULT])
    out = np.zeros(len(days), dtype=np.float64)
    for mm, dd in fixed:
        out[(m == mm) & (d == dd)] = 1.0
    hol_days = set()
    for year in range(int(y.min()), int(y.max()) + 1):
        e = _easter_days(year)
        for off in easter:
            hol_days.add(int(e + off))
    if hol_days:
        out[np.isin(days, np.fromiter(hol_days, dtype=np.int64))] = 1.0
    return out

def build(item_id, unix_sec, temperature=None):
    y, m, d, tod, dow, days = civil_fields(unix_sec)
    hol = holiday_mask(item_id, y, m, d, days)
    sat = ((dow == 5) & (hol == 0)).astype(np.float64)
    sunhol = ((dow == 6) | (hol > 0)).astype(np.float64)
    names = ['cal_saturday', 'cal_sunday_or_holiday']
    feats = [sat, sunhol]
    if temperature is not None:
        t = np.asarray(temperature, dtype=np.float64)
        names += ['cal_hdd15', 'cal_cdd18']
        feats += [np.maximum(0.0, 15.0 - t), np.maximum(0.0, t - 18.0)]
    return (names, np.stack(feats))

def day_type(item_id, unix_sec):
    y, m, d, tod, dow, days = civil_fields(unix_sec)
    hol = holiday_mask(item_id, y, m, d, days)
    dt = np.zeros(len(days), dtype=np.int64)
    dt[dow == 5] = 1
    dt[(dow == 6) | (hol > 0)] = 2
    return (dt, tod, days)

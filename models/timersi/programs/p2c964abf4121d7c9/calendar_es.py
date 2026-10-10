import numpy as np, pandas as pd

def easter(year):
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
    return pd.Timestamp(year=year, month=month, day=day)
FIXED = {(1, 1): 'new_year', (1, 6): 'epiphany', (5, 1): 'labour', (5, 2): 'madrid_dos_mayo', (5, 15): 'san_isidro', (8, 15): 'assumption', (10, 12): 'hispanic', (11, 1): 'all_saints', (11, 9): 'almudena', (12, 6): 'constitution', (12, 8): 'immaculate', (12, 25): 'christmas'}

def holiday_map(years):
    out = {}
    for y in years:
        for (mth, day), name in FIXED.items():
            d = pd.Timestamp(year=y, month=mth, day=day)
            if d.dayofweek == 6 and name not in ('christmas', 'new_year'):
                d = d + pd.Timedelta(days=1)
            out[d] = name
        e = easter(y)
        out[e - pd.Timedelta(days=3)] = 'holy_thursday'
        out[e - pd.Timedelta(days=2)] = 'good_friday'
    return out

def event_features(index):
    index = pd.DatetimeIndex(index)
    years = sorted({index.min().year - 1, *index.year.unique().tolist(), index.max().year + 1})
    hm = holiday_map(years)
    hset = set(hm)
    rows = []
    for d in index:
        wd = d.dayofweek
        is_h = d in hset
        work = wd < 5 and (not is_h)
        prev, nxt = (d - pd.Timedelta(days=1), d + pd.Timedelta(days=1))
        nonwork = lambda x: x.dayofweek >= 5 or x in hset
        bridge = work and nonwork(prev) and nonwork(nxt) and (prev in hset or nxt in hset)
        adj_b = work and (not bridge) and (nxt in hset)
        adj_a = work and (not bridge) and (prev in hset)
        eve = (d.month, d.day) in ((12, 24), (12, 31), (1, 5))
        xmas_break = d.month == 12 and d.day >= 26 or (d.month == 1 and d.day <= 7)
        pre_xmas = d.month == 12 and 10 <= d.day <= 23
        e = easter(d.year)
        holy_week = e - pd.Timedelta(days=7) <= d <= e + pd.Timedelta(days=1)
        august = d.month == 8
        rows.append(dict(date=d, dow=wd, holiday=hm.get(d, ''), is_holiday=is_h, bridge=bridge, adj_before=adj_b and (not is_h), adj_after=adj_a and (not is_h), eve=eve and (not is_h), xmas_break=xmas_break and (not is_h), pre_xmas=pre_xmas and (not is_h), holy_week=holy_week and (not is_h), august=august))
    return pd.DataFrame(rows).set_index('date')

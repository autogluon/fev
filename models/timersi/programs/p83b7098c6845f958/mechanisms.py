import numpy as np
MAXC = 15360

def context_window(hist, cutoff, max_context, limit=MAXC):
    C = int(min(cutoff, max_context, limit))
    C = max(C, 24)
    C -= C % 24
    return C

def spike_winsorize(block, lo_pct=0.5, hi_pct=99.5, protect_hours=168, pad=1.05):
    out = np.array(block, dtype=float, copy=True)
    for d in range(out.shape[0]):
        row = out[d]
        if row.size < 2 * protect_hours:
            continue
        hi = np.percentile(row, hi_pct)
        lo = np.percentile(row, lo_pct)
        recent = row[-protect_hours:]
        hi = max(hi, recent.max() * pad if recent.max() > 0 else hi)
        lo = min(lo, recent.min() * pad if recent.min() < 0 else lo)
        if hi <= lo:
            continue
        out[d] = np.clip(row, lo, hi)
    return out

def robust_asinh(block, protect_hours=0):
    med = np.median(block, axis=1, keepdims=True)
    mad = np.median(np.abs(block - med), axis=1, keepdims=True) * 1.4826
    mad = np.where(mad <= 1e-06, np.std(block, axis=1, keepdims=True) + 1e-06, mad)
    return (np.arcsinh((block - med) / mad), {'med': med.ravel().copy(), 'mad': mad.ravel().copy()})

def inv_robust_asinh(point, quant, params):
    med = params['med'][:, None]
    mad = params['mad'][:, None]
    p = np.sinh(point) * mad + med
    q = np.sinh(quant) * mad[:, :, None] + med[:, :, None]
    return (p, q)
BE_FIXED_HOLIDAYS = {(1, 1), (5, 1), (7, 21), (8, 15), (11, 1), (11, 11), (12, 25)}

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

def be_holiday_flags(timestamps):
    ts = np.asarray(timestamps, dtype='datetime64[h]')
    days = ts.astype('datetime64[D]')
    years = days.astype('datetime64[Y]').astype(int) + 1970
    months = days.astype('datetime64[M]').astype(int) % 12 + 1
    doms = (days - days.astype('datetime64[M]')).astype(int) + 1
    flag = np.zeros(ts.shape[0], dtype=float)
    for mm, dd in BE_FIXED_HOLIDAYS:
        flag[(months == mm) & (doms == dd)] = 1.0
    moving = set()
    for y in np.unique(years):
        e = _easter(int(y))
        for off in (1, 39, 50):
            moving.add(e + np.timedelta64(off, 'D'))
    if moving:
        mov = np.array(sorted(moving), dtype='datetime64[D]')
        flag[np.isin(days, mov)] = 1.0
    return flag

def calendar_block(timestamps):
    ts = np.asarray(timestamps, dtype='datetime64[h]')
    hour = (ts.astype('datetime64[h]').astype(np.int64) % 24).astype(float)
    dow = ((ts.astype('datetime64[D]').astype(np.int64) + 3) % 7).astype(float)
    hol = be_holiday_flags(ts)
    weekend = (dow >= 5).astype(float)
    nonwork = np.maximum(weekend, hol)
    return (np.stack([np.sin(2 * np.pi * hour / 24.0), np.cos(2 * np.pi * hour / 24.0), np.sin(2 * np.pi * dow / 7.0), np.cos(2 * np.pi * dow / 7.0), nonwork]), ['cal_hour_sin', 'cal_hour_cos', 'cal_dow_sin', 'cal_dow_cos', 'cal_nonworking'])

def find_covariate_indices(names):
    lower = [str(n).lower() for n in names]
    gi = next((i for i, n in enumerate(lower) if 'generation' in n), None)
    li = next((i for i, n in enumerate(lower) if 'load' in n), None)
    return (gi, li)

def scarcity_block(known, gi, li, origin, hour_of_day, lookback_days=365):
    g = np.asarray(known[gi], dtype=float)
    l = np.asarray(known[li], dtype=float)
    denom = np.where(np.abs(g) < 1e-06, np.nan, g)
    ratio = l / denom
    ratio = np.where(np.isfinite(ratio), ratio, np.nanmedian(ratio[np.isfinite(ratio)]) if np.isfinite(ratio).any() else 1.0)
    T = ratio.shape[0]
    start = max(0, int(origin) - lookback_days * 24)
    pct = np.full(T, 0.5)
    z = np.zeros(T)
    for h in range(24):
        idx = np.arange(T)[hour_of_day == h]
        if idx.size == 0:
            continue
        ref_idx = idx[(idx >= start) & (idx < origin)]
        if ref_idx.size < 30:
            ref_idx = idx[idx < origin]
        if ref_idx.size < 5:
            continue
        ref = np.sort(ratio[ref_idx])
        pct[idx] = np.searchsorted(ref, ratio[idx], side='left') / float(ref.size)
        mu, sd = (ref.mean(), ref.std())
        z[idx] = (ratio[idx] - mu) / (sd if sd > 1e-09 else 1.0)
    return (np.stack([pct, np.clip(z, -6.0, 6.0)]), ['scarcity_pct', 'scarcity_z'])

def hour_of_day(timestamps):
    ts = np.asarray(timestamps, dtype='datetime64[h]')
    return (ts.astype(np.int64) % 24).astype(int)

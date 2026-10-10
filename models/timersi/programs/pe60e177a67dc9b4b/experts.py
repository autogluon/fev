import numpy as np
QLEV = np.arange(1, 10) / 10.0
DAY = 288

def _weekday_of(ts64):
    return int((np.datetime64(ts64, 'D').astype('int64') + 3) % 7)

def daytype_quantiles(hist, L, H, first_future_ts, n_match_week=4, slot_win=2, day=DAY):
    wd0 = _weekday_of(first_future_ts)
    target_is_weekday = wd0 < 5
    want = n_match_week * (5 if target_is_weekday else 2)
    ks, k = ([], 1)
    while len(ks) < want and L - k * day >= 0:
        wd = (wd0 - k) % 7
        if (wd < 5) == target_is_weekday:
            ks.append(k)
        k += 1
        if k > 400:
            break
    if len(ks) < 4:
        return None
    hh = np.arange(H)
    cols = []
    for kk in ks:
        for d in range(-slot_win, slot_win + 1):
            idx = L + hh + d - kk * day
            idx = np.clip(idx, 0, L - 1)
            cols.append(hist[:, idx])
    S = np.stack(cols, axis=1)
    q = np.quantile(S, QLEV, axis=1)
    return np.moveaxis(q, 0, -1)

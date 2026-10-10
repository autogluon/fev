import numpy as np
LOG_THRESH = 0.85
REF_DAYS = 14
WIN = 8
NEED = 6
MIN_SEG_HOURS = 336
MIN_HIST_DAYS = 120
PERSIST_DAYS = 60

def detect_regime_start(y, observed=None, period=24):
    y = np.asarray(y, dtype=float)
    n = y.size
    if observed is not None:
        obs = np.asarray(observed, dtype=bool)
    else:
        obs = np.isfinite(y)
    nd = n // period
    if nd < MIN_HIST_DAYS:
        return 0
    tail = n - nd * period
    v = np.where(obs & np.isfinite(y), y, np.nan)[tail:].reshape(nd, period)
    with np.errstate(invalid='ignore'):
        daily = np.nanmean(v, axis=1)
    good = np.isfinite(daily) & (daily > 0)
    if good.sum() < MIN_HIST_DAYS:
        return 0
    ref = np.median(daily[good][-REF_DAYS:])
    if not np.isfinite(ref) or ref <= 0:
        return 0
    with np.errstate(divide='ignore', invalid='ignore'):
        dev = np.abs(np.log(np.where(daily > 0, daily, np.nan) / ref))
    bad = np.where(np.isfinite(dev), dev > LOG_THRESH, False)
    brk = None
    for b in range(nd - REF_DAYS, WIN - 1, -1):
        if bad[b - WIN:b].sum() < NEED:
            continue
        before = daily[max(0, b - PERSIST_DAYS):b]
        before = before[np.isfinite(before) & (before > 0)]
        if before.size < WIN:
            continue
        if abs(np.log(np.median(before) / ref)) <= LOG_THRESH:
            continue
        brk = b
        break
    if brk is None:
        return 0
    while brk < nd - 1 and bad[brk]:
        brk += 1
    seg_days = nd - brk
    if seg_days < MIN_SEG_HOURS // period:
        return 0
    start = tail + brk * period
    start = min(start, n - MIN_SEG_HOURS)
    return max(int(start), 0)

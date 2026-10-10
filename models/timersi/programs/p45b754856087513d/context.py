import numpy as np

def active_start(y, min_keep=180, gap=14):
    y = np.asarray(y, float)
    n = len(y)
    nz = np.flatnonzero(y > 0)
    if len(nz) == 0:
        return 0
    start = int(nz[0])
    jumps = np.flatnonzero(np.diff(nz) >= gap)
    if len(jumps):
        start = int(nz[jumps[-1] + 1])
    if n - start < min_keep:
        start = max(0, n - min_keep)
    return start

def impute_isolated_zeros(y, dates_dow, max_run=3):
    y = np.asarray(y, float).copy()
    n = len(y)
    zero = y <= 0
    if not zero.any():
        return (y, np.zeros(n, bool))
    filled = np.zeros(n, bool)
    i = 0
    while i < n:
        if zero[i]:
            j = i
            while j < n and zero[j]:
                j += 1
            if 0 < i and j < n and (j - i <= max_run):
                for t in range(i, j):
                    lo, hi = (max(0, t - 28), min(n, t + 29))
                    win = y[lo:hi]
                    same = win[(dates_dow[lo:hi] == dates_dow[t]) & (win > 0)]
                    nearby = win[win > 0]
                    if len(same) >= 3:
                        y[t] = float(np.median(same))
                    elif len(nearby) >= 3:
                        y[t] = float(np.median(nearby))
                    filled[t] = y[t] > 0
            i = j
        else:
            i += 1
    return (y, filled)

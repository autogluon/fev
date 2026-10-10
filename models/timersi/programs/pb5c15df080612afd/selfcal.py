import numpy as np
import counter
QL = np.arange(1, 10) / 10.0

def _origins(days_arr, L, R):
    d = days_arr[:L]
    last = d[-1]
    out = []
    for r in range(1, R + 1):
        end = np.searchsorted(d, last - r + 1, side='left')
        if end >= 48 + 24 and end + 24 <= L:
            out.append(int(end))
    return out

def error_profile(hist, hour, dow, days_arr, L, H, R=8, nsim=120, half_lives=(4.0, 8.0, 16.0), n_days_back=120):
    N = hist.shape[0]
    if H != 24:
        return None
    ors = _origins(days_arr, L, R)
    if len(ors) < 3:
        return None
    real = np.zeros((N, H))
    imp = np.zeros((N, H))
    cnt = 0
    for o in ors:
        fh, fd = (hour[o:o + H], dow[o:o + H])
        if not np.array_equal(fh, np.arange(24)):
            continue
        truth = hist[:, o:o + H]
        ok_origin = False
        for i in range(N):
            x = hist[i, :o]
            if len(x) < 24 * 10 or not np.all(np.isfinite(x)):
                continue
            try:
                _, q, _ = counter.forecast_mix(x, hour[:o], dow[:o], days_arr[:o], fh, fd, nsim=nsim, seed=1000 + i, half_lives=half_lives, n_days_back=n_days_back)
            except Exception:
                continue
            med = q[:, 4]
            real[i] += np.abs(truth[i] - med)
            imp[i] += np.abs(q - med[:, None]).mean(axis=1)
            ok_origin = True
        if ok_origin:
            cnt += 1
    if cnt == 0:
        return None
    return (real / cnt, imp / cnt)

def inflation(real, imp, lo=0.7, hi=3.0, pool=0.5):
    eps = 1e-06
    f_t = real / np.maximum(imp, eps)
    f_g = (real.mean(axis=1) / np.maximum(imp.mean(axis=1), eps))[:, None]
    f = pool * f_g + (1.0 - pool) * f_t
    return np.clip(np.nan_to_num(f, nan=1.0, posinf=hi, neginf=lo), lo, hi)

import numpy as np

def _ppf(p):
    a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239]
    b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572]
    c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783]
    d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416]
    pl, ph = (0.02425, 1 - 0.02425)
    if p < pl:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > ph:
        q = np.sqrt(-2 * np.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)

def ewma_level(X, halflife):
    a = 1.0 - 0.5 ** (1.0 / float(halflife))
    out = np.empty_like(X)
    s = X[:, 0].copy()
    out[:, 0] = s
    for t in range(1, X.shape[1]):
        s += a * (X[:, t] - s)
        out[:, t] = s
    return out

def backtest_error_quantiles(X, H, levels, halflife=6.0, stride=12, max_origins=600, min_warmup=200, n_blocks=8):
    D, L = X.shape
    last = L - H - 1
    if last < min_warmup:
        return None
    origins = np.arange(last, min_warmup - 1, -stride)[:max_origins][::-1]
    if origins.size < 12:
        return None
    lev = ewma_level(X, halflife)[:, origins]
    idx = origins[:, None] + np.arange(1, H + 1)[None, :]
    err = X[:, idx] - lev[:, :, None]
    nb = max(1, min(n_blocks, H))
    bl = H // nb
    use = bl * nb
    eb = err[:, :, :use].reshape(D, origins.size, nb, bl)
    eb = eb.transpose(0, 2, 1, 3).reshape(D, nb, -1)
    qs = np.quantile(eb, levels, axis=2)
    qs = np.repeat(qs, bl, axis=2)
    if use < H:
        qs = np.concatenate([qs, np.repeat(qs[:, :, -1:], H - use, axis=2)], axis=2)
    return qs
SQRT2 = 1.2815515655446004

def calibrate_spread(quantiles, history, H, weight=0.4, floor=0.2, ceiling=1.0, halflives=(6.0, 24.0, 96.0), stride=12, max_origins=600, n_blocks=8, ref_p=0.2, lower_extra=0.92):
    lows, ups = ([], [])
    for hl in halflives:
        qs = backtest_error_quantiles(history, H, [ref_p, 0.5, 1.0 - ref_p], halflife=hl, stride=stride, max_origins=max_origins, n_blocks=n_blocks)
        if qs is None:
            continue
        lows.append(np.maximum(qs[1] - qs[0], 1e-09))
        ups.append(np.maximum(qs[2] - qs[1], 1e-09))
    if not lows:
        return (quantiles, None)
    eps = 1e-09
    gauss = SQRT2 / max(abs(_ppf(1.0 - ref_p)), 1e-06)
    loc_low = np.median(lows, axis=0) * gauss
    loc_up = np.median(ups, axis=0) * gauss
    med = quantiles[:, :, 4:5]
    nat_low = np.maximum(med[:, :, 0] - quantiles[:, :, 0], eps)
    nat_up = np.maximum(quantiles[:, :, 8] - med[:, :, 0], eps)
    f_low = np.clip((loc_low / nat_low) ** weight, floor, ceiling)[:, :, None] * lower_extra
    f_up = np.clip((loc_up / nat_up) ** weight, floor, ceiling)[:, :, None]
    dev = quantiles - med
    out = med + np.where(dev < 0.0, f_low * dev, f_up * dev)
    out = np.sort(out, axis=2)
    diag = {'f_low': float(np.mean(f_low)), 'f_up': float(np.mean(f_up))}
    return (out, diag)

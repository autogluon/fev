import numpy as np
QLEV = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
MIN_BACKTEST_CONTEXT = 26
OFFSETS = (0, 26)
MAX_LEVEL_SHIFT = 0.1
WIDTH_RANGE = (0.8, 1.25)
STRENGTH = 0.5
PRIOR_WLO = 0.92
PRIOR_WHI = 1.0

def plan_origins(L, H, max_extra=2):
    out = []
    for extra in OFFSETS:
        o = H + extra
        c = L - o
        if c < 1:
            continue
        out.append(int(o))
        if len(out) >= max_extra:
            break
    return out

def pinball(truth, q):
    d = truth[..., None] - q
    return np.maximum(QLEV * d, (QLEV - 1) * d)

def measure(records):
    usable = [r for r in records if r['context'] >= MIN_BACKTEST_CONTEXT and np.isfinite(r['q']).all() and np.isfinite(r['truth']).all()]
    diag = {'n_records': len(records), 'n_usable': len(usable)}
    if not usable:
        diag['reason'] = 'no backtest context >= %d; weak prior only' % MIN_BACKTEST_CONTEXT
        return {'shift': 0.0, 'wlo': PRIOR_WLO, 'whi': PRIOR_WHI, 'diag': diag}
    shifts = []
    for r in usable:
        med = r['q'][:, 4]
        ok = (med > 0) & (r['truth'] > 0)
        if ok.sum() >= 8:
            shifts.append(float(np.median(np.log(r['truth'][ok] / med[ok]))))
    shift = 0.0
    if shifts:
        if len(shifts) == 1 or min(shifts) > 0 or max(shifts) < 0:
            shift = float(np.clip(np.median(shifts), -MAX_LEVEL_SHIFT, MAX_LEVEL_SHIFT))
    diag['shifts'] = shifts

    def loss(wlo, whi):
        tot = 0.0
        for r in usable:
            q = r['q']
            med = q[:, 4:5]
            w = np.ones(9)
            w[:4] = wlo
            w[5:] = whi
            qq = np.sort(med + (q - med) * w[None, :], axis=1) * np.exp(shift)
            tot += float(pinball(r['truth'], qq).mean())
        return tot
    grid = np.round(np.arange(WIDTH_RANGE[0], WIDTH_RANGE[1] + 1e-09, 0.05), 3)
    best, bl = ((1.0, 1.0), loss(1.0, 1.0))
    for a in grid:
        for b in grid:
            v = loss(a, b)
            if v < bl - 1e-12:
                bl, best = (v, (float(a), float(b)))
    diag['raw_width'] = best
    diag['coverage'] = [float(np.mean([np.mean(r['truth'][:, None] <= r['q'], axis=0)[k] for r in usable])) for k in range(9)]
    return {'shift': STRENGTH * shift, 'wlo': 1.0 + STRENGTH * (best[0] - 1.0), 'whi': 1.0 + STRENGTH * (best[1] - 1.0), 'diag': diag}

def apply_calibration(point, quantiles, cal):
    f = float(np.exp(cal['shift']))
    q = np.asarray(quantiles, float) * f
    p = np.asarray(point, float) * f
    wlo, whi = (float(cal['wlo']), float(cal['whi']))
    if abs(wlo - 1.0) > 1e-09 or abs(whi - 1.0) > 1e-09:
        med = q[..., 4:5]
        w = np.ones(9)
        w[:4] = wlo
        w[5:] = whi
        q = med + (q - med) * w
    q = np.sort(q, axis=-1)
    return (p, q)

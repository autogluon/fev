import numpy as np
QL = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
GRID = (0.0, 0.5, 0.75, 1.0, 1.25, 1.5)
SHRINK = 0.5
LO, HI = (0.6, 1.6)

def scaled_pinball(y, q, scale):
    d = y[:, None] - q
    return float(np.maximum(QL * d, (QL - 1) * d).mean() * 2.0 / scale)

def strength(records):
    use = [r for r in records if np.abs(r['log_cal']).max() > 1e-06 and r['scale'] > 0]
    if not use:
        return (1.0, {'n_origins': 0, 'reason': 'no informative auxiliary origin'})
    curve = {}
    for w in GRID:
        ratios = []
        for r in use:
            fac = np.exp(np.clip(r['log_cal'] * w, -0.25, 0.25))
            med = r['quant'][:, 4]
            q = np.sort((r['quant'] + (r['point'] - med)[:, None]) * fac[:, None], axis=1)
            base = scaled_pinball(r['y'], r['quant'], r['scale'])
            if base <= 0:
                continue
            ratios.append(scaled_pinball(r['y'], q, r['scale']) / base)
        curve[w] = float(np.mean(ratios)) if ratios else np.nan
    valid = {w: v for w, v in curve.items() if np.isfinite(v)}
    if not valid:
        return (1.0, {'n_origins': len(use), 'reason': 'no finite replay loss'})
    best = min(valid, key=valid.get)
    w = float(np.clip(1.0 + SHRINK * (best - 1.0), LO, HI))
    return (w, {'n_origins': len(use), 'grid': {str(k): v for k, v in curve.items()}, 'argmin': best, 'applied': w})

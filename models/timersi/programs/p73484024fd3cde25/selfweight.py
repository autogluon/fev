import numpy as np
from corrections import apply_correction, scaled_pinball
GRID_LVL = (0.0, 0.5, 1.0, 1.25)
GRID_CAL = (0.0, 0.5, 1.0, 1.5)
GRID_Q = (0.0, 0.5, 1.0, 1.5)
SHRINK = 0.5
MIN_RECORDS = 40

def collect(origin, outputs_by_origin, hist, obs):
    recs = []
    for org, out in outputs_by_origin:
        p = np.asarray(out['point'], float).reshape(org.N, org.H)
        q = np.asarray(out['quantiles'], float).reshape(org.N, org.H, 9)
        for d in range(org.N):
            y = hist[d][org.end:org.end + org.H]
            o = obs[d][org.end:org.end + org.H]
            ctx = org.hist[d][org.obs[d]]
            ctx = ctx[np.isfinite(ctx)]
            if ctx.size < 2 or not np.all(o) or (not np.all(np.isfinite(y))):
                continue
            sc = float(np.mean(np.abs(np.diff(ctx))))
            if not np.isfinite(sc) or sc <= 1e-09:
                continue
            if not (np.all(np.isfinite(p[d])) and np.all(np.isfinite(q[d]))):
                continue
            recs.append((org, d, p[d], q[d], y, sc))
    return recs

def _loss(recs, w_lvl, w_cal, w_q):
    vals = []
    for org, d, p, q, y, sc in recs:
        pc, qc = apply_correction(org, d, p, q, w_lvl, w_cal, w_q)
        base = scaled_pinball(y, q, sc)
        if base <= 0:
            continue
        vals.append(scaled_pinball(y, qc, sc) / base)
    return float(np.mean(vals)) if vals else np.nan

def choose(recs, prior=(1.0, 1.0, 1.0)):
    info = {'n_records': len(recs)}
    if len(recs) < MIN_RECORDS:
        info['reason'] = 'too few scorable replays'
        return (prior, info)
    w = list(prior)
    curves = {}
    for rnd in range(2):
        for idx, grid in ((0, GRID_LVL), (1, GRID_CAL), (2, GRID_Q)):
            cur = {}
            for g in grid:
                trial = list(w)
                trial[idx] = g
                cur[g] = _loss(recs, *trial)
            fin = {k: v for k, v in cur.items() if np.isfinite(v)}
            if not fin:
                continue
            w[idx] = min(fin, key=fin.get)
            if rnd == 1:
                curves[['lvl', 'cal', 'q'][idx]] = {str(k): round(v, 6) for k, v in cur.items()}
    final = tuple((float(np.clip(p + SHRINK * (b - p), 0.0, 1.5)) for p, b in zip(prior, w)))
    info.update({'argmin': w, 'applied': list(final), 'curves': curves, 'loss_prior': _loss(recs, *prior), 'loss_argmin': _loss(recs, *w)})
    return (final, info)

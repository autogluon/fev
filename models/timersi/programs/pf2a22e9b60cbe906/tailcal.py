import numpy as np
from corrections import Origin, apply_correction
QL = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
GRID_LO = (1.0, 1.25, 1.5, 1.75, 2.0)
GRID_UP = (1.0,)
SHRINK = 0.5
LO_CAP = (1.0, 1.8)
UP_CAP = (1.0, 1.0)
MIN_SAMPLES = 13

def _tail_pinball(y, q, scale, js):
    d = y[:, None] - q[:, js]
    tau = QL[js]
    return float(np.maximum(tau * d, (tau - 1) * d).mean() * 2.0 / scale)

def _scale_tail(q, m_lo, m_up):
    med = q[:, 4]
    dev = q - med[:, None]
    out = med[:, None] + np.where(dev < 0, dev * m_lo, dev * m_up)
    return np.sort(out, axis=1)

def collect_replays(state, outputs, cal_min_context):
    recs = []
    H = state['H']
    for i, m in enumerate(state['case_meta']):
        if m['kind'] != 'auxiliary' or i >= len(outputs) or outputs[i] is None:
            continue
        end = m['end']
        p = np.asarray(outputs[i]['point'], float).reshape(state['N'], H)
        q = np.asarray(outputs[i]['quantiles'], float).reshape(state['N'], H, 9)
        org = Origin(state['hist'], state['obs'], state['ts'], end, H, want_cal=end >= cal_min_context)
        for d in range(state['N']):
            y = state['hist'][d][end:end + H]
            o = state['obs'][d][end:end + H]
            ctx = state['hist'][d][:end][state['obs'][d][:end]]
            ctx = ctx[np.isfinite(ctx)]
            if ctx.size < 2 or not np.all(o) or (not np.all(np.isfinite(y))):
                continue
            sc = float(np.mean(np.abs(np.diff(ctx))))
            if not np.isfinite(sc) or sc <= 1e-09:
                continue
            if not (np.all(np.isfinite(p[d])) and np.all(np.isfinite(q[d]))):
                continue
            pc, qc = apply_correction(org, d, p[d], q[d], 1.0, 1.0, 1.0, nonneg=state['nonneg'])
            recs.append((y, qc, sc))
    return recs

def choose_multipliers(recs):
    info = {'n_samples': int(sum((len(r[0]) for r in recs)))}
    if info['n_samples'] < MIN_SAMPLES:
        info['reason'] = 'too few matured replay leads'
        return (1.0, 1.0, info)

    def pooled(m_lo, m_up, js):
        vals = [_tail_pinball(y, _scale_tail(q, m_lo, m_up), sc, js) for y, q, sc in recs]
        return float(np.mean(vals))
    lo_curve = {m: pooled(m, 1.0, np.arange(0, 4)) for m in GRID_LO}
    up_curve = {m: pooled(1.0, m, np.arange(5, 9)) for m in GRID_UP}
    b_lo = min(lo_curve, key=lo_curve.get)
    b_up = min(up_curve, key=up_curve.get)
    m_lo = float(np.clip(1.0 + SHRINK * (b_lo - 1.0), *LO_CAP))
    m_up = float(np.clip(1.0 + SHRINK * (b_up - 1.0), *UP_CAP))
    info.update({'lo_curve': {str(k): round(v, 5) for k, v in lo_curve.items()}, 'up_curve': {str(k): round(v, 5) for k, v in up_curve.items()}, 'applied_lo': m_lo, 'applied_up': m_up})
    return (m_lo, m_up, info)

def apply_to_final(quant, m_lo, m_up, protect):
    q = np.asarray(quant, float).copy()
    med = q[:, 4]
    dev = q - med[:, None]
    row_lo = np.where(protect, 1.0, m_lo)
    row_up = np.where(protect, 1.0, m_up)
    out = med[:, None] + np.where(dev < 0, dev * row_lo[:, None], dev * row_up[:, None])
    return np.sort(out, axis=1)

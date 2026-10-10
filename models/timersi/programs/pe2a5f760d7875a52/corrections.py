import numpy as np
from calendar_ops import N_FEATURES, applied_design, fit_calendar, week_features
ANCHOR_MAX = 0.85
ANCHOR_REF = 25.0
ANCHOR_WIN = 4
TREND_MAX = 0.45
TREND_SCALE = 260.0
CAL_CLIP = 0.15
Q_LOW = 0.8
Q_HIGH = 0.9
Q_SHIFT = 0.06
FAN = np.where(np.arange(9) < 4, Q_LOW, np.where(np.arange(9) > 4, Q_HIGH, 1.0))
QL = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

def anchor_value(y, obs, win=ANCHOR_WIN):
    v = np.asarray(y, float)[np.asarray(obs, bool) & np.isfinite(y)]
    if v.size == 0:
        return None
    pos = v[v > 0]
    use = pos if pos.size >= 2 else v
    return float(np.median(use[-min(win, use.size):]))

def calendar_log(hist_d, obs_d, ts_past, ts_future):
    H = len(ts_future)
    Xp = week_features(ts_past) if len(ts_past) else np.zeros((0, N_FEATURES))
    beta, w = fit_calendar(np.where(obs_d, hist_d, np.nan), Xp)
    Xf = applied_design(week_features(ts_future))
    return np.clip(Xf @ beta * w, -CAL_CLIP, CAL_CLIP)

class Origin:

    def __init__(self, hist, obs, ts, end, H, want_cal=True):
        self.end, self.H = (end, H)
        self.L = end
        self.hist = hist[:, :end]
        self.obs = obs[:, :end]
        self.N = hist.shape[0]
        self.anchor = [anchor_value(self.hist[d], self.obs[d]) for d in range(self.N)]
        ts_p = list(ts[:end])
        ts_f = list(ts[end:end + H])
        if want_cal:
            self.cal = np.stack([calendar_log(self.hist[d], self.obs[d], ts_p, ts_f) for d in range(self.N)])
        else:
            self.cal = np.zeros((self.N, H))
        self.alpha0 = ANCHOR_MAX * ANCHOR_REF / (ANCHOR_REF + end)
        self.gamma0 = min(TREND_MAX, end / TREND_SCALE)

def apply_correction(origin, d, point, quant, w_lvl, w_cal, w_q, nonneg=True):
    H = origin.H
    h = np.arange(H, dtype=float)
    hc = h - h.mean()
    den = float((hc ** 2).sum()) or 1.0
    p = np.asarray(point, float)
    q = np.asarray(quant, float)
    b = float((hc * p).sum() / den)
    a = float(p.mean() - b * h.mean())
    shape = p - (a + b * h)
    alpha = w_lvl * origin.alpha0
    gamma = 1.0 - w_lvl * (1.0 - origin.gamma0)
    anc = origin.anchor[d]
    lvl = a if anc is None else (1.0 - alpha) * a + alpha * anc
    p2 = lvl + gamma * b * h + shape
    cal = np.clip(origin.cal[d] * w_cal, -CAL_CLIP, CAL_CLIP)
    fac = np.exp(cal)
    med = q[:, 4]
    iqr = q[:, 7] - q[:, 1]
    fan = 1.0 + w_q * (FAN - 1.0)
    q2 = med[:, None] + fan * (q - med[:, None]) + (p2 - med)[:, None] + w_q * Q_SHIFT * iqr[:, None]
    q2 = q2 * fac[:, None]
    p3 = p2 * fac
    if nonneg:
        p3 = np.maximum(p3, 0.0)
        q2 = np.maximum(q2, 0.0)
    return (p3, np.sort(q2, axis=1))

def scaled_pinball(y, q, scale):
    d = np.asarray(y, float)[:, None] - np.asarray(q, float)
    return float(np.maximum(QL * d, (QL - 1) * d).mean() * 2.0 / scale)

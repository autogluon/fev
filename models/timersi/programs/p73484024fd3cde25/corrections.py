import numpy as np
from calendar_ops import N_FEATURES, applied_design, fit_calendar, fixed_design, week_features
from holiday_prior import LO_WIDEN_FRAC, PRIOR_REF, UP_WIDEN, evidence_lambda, fixed_prior_log
from unseen_event import FIXED, MOVABLE, event_mass, unseen_weight
ANCHOR_MAX = 0.85
ANCHOR_REF = 25.0
ANCHOR_WIN = 4
TREND_MAX = 0.45
TREND_SCALE = 260.0
CAL_CLIP = 0.15
DEC_CLIP = 0.2
WIDEN = 1.0
Q_LOW = 0.8
Q_HIGH = 0.9
Q_SHIFT = 0.06
FAN = np.where(np.arange(9) < 4, Q_LOW, np.where(np.arange(9) > 4, Q_HIGH, 1.0))
QL = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
EVENT_SKIP = None

def anchor_value(y, obs, win=ANCHOR_WIN, event_load=None):
    y = np.asarray(y, float)
    keep = np.asarray(obs, bool) & np.isfinite(y)
    idx = np.where(keep)[0]
    if idx.size == 0:
        return None
    if event_load is not None and EVENT_SKIP is not None:
        ordinary = idx[np.asarray(event_load, float)[idx] <= EVENT_SKIP]
        pos_ord = ordinary[y[ordinary] > 0]
        if pos_ord.size >= max(2, win // 2):
            return float(np.median(y[pos_ord[-min(win, pos_ord.size):]]))
    v = y[idx]
    pos = v[v > 0]
    use = pos if pos.size >= 2 else v
    return float(np.median(use[-min(win, use.size):]))

def calendar_log(hist_d, obs_d, ts_past, ts_future, dec_ok):
    Xp = week_features(ts_past) if len(ts_past) else np.zeros((0, N_FEATURES))
    beta, w = fit_calendar(np.where(obs_d, hist_d, np.nan), Xp)
    Xw = week_features(ts_future)
    reg = np.clip(applied_design(Xw) @ beta * w, -CAL_CLIP, CAL_CLIP)
    dec = np.clip(fixed_design(Xw) @ beta * w, -DEC_CLIP, DEC_CLIP) if dec_ok else np.zeros(len(ts_future))
    return (reg, dec)

class Origin:

    def __init__(self, hist, obs, ts, end, H, want_cal=True):
        self.end, self.H = (end, H)
        self.L = end
        self.hist = hist[:, :end]
        self.obs = obs[:, :end]
        self.N = hist.shape[0]
        ts_p = list(ts[:end])
        Xp = week_features(ts_p) if end else np.zeros((0, N_FEATURES))
        load = Xp[:, MOVABLE] if len(Xp) else np.zeros(0)
        self.event_load = load
        self.anchor = [anchor_value(self.hist[d], self.obs[d], event_load=load) for d in range(self.N)]
        ts_f = list(ts[end:end + H])
        self.fixed_mass = event_mass(ts_p, FIXED)
        self.dec_ok = bool(want_cal and self.fixed_mass >= 1.0)
        self.unseen = unseen_weight(ts_p, ts_f)
        self.lam = evidence_lambda(self.fixed_mass)
        self.prior = fixed_prior_log(ts_f)
        Xf_w = week_features(ts_f) if H else np.zeros((0, N_FEATURES))
        mov_cold = Xf_w[:, MOVABLE] if event_mass(ts_p, MOVABLE) < 1.0 else np.zeros(H)
        self.unseen_mov = np.clip(mov_cold, 0.0, 1.0)
        if want_cal:
            pairs = [calendar_log(self.hist[d], self.obs[d], ts_p, ts_f, self.dec_ok) for d in range(self.N)]
            self.cal = np.stack([p[0] for p in pairs])
            self.dec = np.stack([p[1] for p in pairs])
        else:
            self.cal = np.zeros((self.N, H))
            self.dec = np.zeros((self.N, H))
        self.alpha0 = ANCHOR_MAX * ANCHOR_REF / (ANCHOR_REF + end)
        self.gamma0 = min(TREND_MAX, end / TREND_SCALE)

def _already_there(point, path):
    H = len(path)
    u = np.asarray(path, float)
    if float(np.abs(u).max()) < 1e-08:
        return 0.0
    h = np.arange(H, dtype=float)
    A = np.c_[np.ones(H), h - h.mean()]
    P = A @ np.linalg.pinv(A)
    uc = u - P @ u
    den = float(uc @ uc)
    if den < 1e-10:
        return 0.0
    lg = np.log(np.maximum(np.asarray(point, float), 1e-06))
    c = float(uc @ (lg - P @ lg) / den)
    return float(min(max(c, 0.0), 1.0))

def apply_correction(origin, d, point, quant, w_lvl, w_cal, w_q, nonneg=True, w_dec=1.0, w_widen=None, w_anc=None, w_trend=None):
    H = origin.H
    h = np.arange(H, dtype=float)
    hc = h - h.mean()
    den = float((hc ** 2).sum()) or 1.0
    p = np.asarray(point, float)
    q = np.asarray(quant, float)
    b = float((hc * p).sum() / den)
    a = float(p.mean() - b * h.mean())
    shape = p - (a + b * h)
    wa = w_lvl if w_anc is None else w_anc
    wt = w_lvl if w_trend is None else w_trend
    alpha = wa * origin.alpha0
    gamma = 1.0 - wt * (1.0 - origin.gamma0)
    anc = origin.anchor[d]
    lvl = a if anc is None else (1.0 - alpha) * a + alpha * anc
    p2 = lvl + gamma * b * h + shape
    cal = np.clip(origin.cal[d] * w_cal, -CAL_CLIP, CAL_CLIP)
    dec = getattr(origin, 'dec', np.zeros(H))[d] * w_dec
    if np.abs(dec).max() > 1e-08:
        dec = dec * (1.0 - _already_there(p, dec))
    lam = getattr(origin, 'lam', 1.0)
    prior = getattr(origin, 'prior', np.zeros(H)) * w_dec
    dec_total = np.clip(dec + (1.0 - lam) * prior, -DEC_CLIP, DEC_CLIP)
    if np.abs(dec_total).max() > 1e-08:
        cal = cal + dec_total
    fac = np.exp(cal)
    med = q[:, 4]
    iqr = q[:, 7] - q[:, 1]
    fan = 1.0 + w_q * (FAN - 1.0)
    q2 = med[:, None] + fan * (q - med[:, None]) + (p2 - med)[:, None] + w_q * Q_SHIFT * iqr[:, None]
    q2 = q2 * fac[:, None]
    uw = getattr(origin, 'unseen_mov', getattr(origin, 'unseen', np.zeros(H)))
    w_widen = WIDEN if w_widen is None else w_widen
    if w_widen > 0 and float(uw.max()) > 1e-08:
        m2 = q2[:, 4]
        q2 = m2[:, None] + (1.0 + w_widen * uw)[:, None] * (q2 - m2[:, None])
    wasym = (1.0 - lam) * np.clip(getattr(origin, 'prior', np.zeros(H)) / PRIOR_REF, 0.0, 1.0)
    if float(wasym.max()) > 1e-08:
        m3 = q2[:, 4]
        dev = q2 - m3[:, None]
        up = 1.0 + UP_WIDEN * wasym
        lo = 1.0 + LO_WIDEN_FRAC * UP_WIDEN * wasym
        q2 = m3[:, None] + np.where(dev > 0, dev * up[:, None], dev * lo[:, None])
    p3 = p2 * fac
    if nonneg:
        p3 = np.maximum(p3, 0.0)
        q2 = np.maximum(q2, 0.0)
    return (p3, np.sort(q2, axis=1))

def scaled_pinball(y, q, scale):
    d = np.asarray(y, float)[:, None] - np.asarray(q, float)
    return float(np.maximum(QL * d, (QL - 1) * d).mean() * 2.0 / scale)

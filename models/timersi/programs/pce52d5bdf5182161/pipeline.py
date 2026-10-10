import datetime
import numpy as np
try:
    from retail_model import fit_predict, effective_open
except ImportError:
    from .retail_model import fit_predict, effective_open
QL = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
ZQ = np.array([-1.2815515655446004, -0.8416212335729143, -0.5244005127080407, -0.2533471031357997, 0.0, 0.2533471031357997, 0.5244005127080407, 0.8416212335729143, 1.2815515655446004])
CORR_W_BASE = 0.5
CORR_W_GAIN = 0.2
CORR_W_HALF = 40.0
SHAPE_HALF = 26.0
SHAPE_MAX = 0.6
SLOPE_HALF = 26.0
GUARD_TAU = 10.0
ASYM = np.where(np.arange(9) < 4, 0.88, np.where(np.arange(9) > 4, 1.12, 1.0))

def _as2d(x):
    a = np.asarray(x, dtype=float)
    if a.ndim == 1:
        a = a[None, :]
    return a

def _calendar(timestamps):
    out = []
    for t in timestamps:
        s = str(t)[:19]
        try:
            d = datetime.datetime.fromisoformat(s)
        except ValueError:
            d = datetime.datetime.strptime(s[:10], '%Y-%m-%d')
        out.append(d.timetuple().tm_yday / 365.25)
    return np.asarray(out, float)

def preprocess(view, card):
    L = int(view['cutoff_index'])
    H = int(view['horizon'])
    D = len(view['target_ids'])
    hist = _as2d(view['target_history'])
    obs = np.asarray(view['target_observed'], dtype=bool)
    if obs.ndim == 1:
        obs = obs[None, :]
    known = {}
    kf = view.get('known_features')
    if kf is not None and len(kf):
        kf = _as2d(kf)
        for nm, row in zip(view['known_names'], kf):
            known[nm] = np.asarray(row, float)
    ones = np.ones(L + H)
    open_d = np.clip(known.get('Open', ones), 0.0, 1.0) * 7.0
    promo_d = np.clip(known.get('Promo', np.zeros(L + H)), 0.0, 1.0) * 7.0
    promo_d = np.minimum(promo_d, open_d)
    school_d = np.clip(known.get('SchoolHoliday', np.zeros(L + H)), 0.0, 7.0)
    state_d = np.clip(known.get('StateHoliday', np.zeros(L + H)), 0.0, 7.0)
    nonneg = bool(np.all(hist[np.isfinite(hist)] >= 0.0)) if np.isfinite(hist).any() else False
    obs_any = obs.any(axis=0) if obs.size else np.zeros(L, bool)
    eff, typ = effective_open(open_d, state_d, L, obs_any[:L] & (open_d[:L] > 0.3))
    state = {'N': D, 'H': H, 'L': L, 'hist': hist, 'obs': obs, 'open_d': open_d, 'promo_d': promo_d, 'school_d': school_d, 'state_d': state_d, 'woy': _calendar(view['timestamps']), 'nonneg': nonneg, 'item_id': view.get('item_id'), 'exposure_future': eff[L:], 'typ_open': typ}
    return (view, state)

def engineer(prepared, card, state):
    L, H = (state['L'], state['H'])
    preds = np.zeros((state['N'], H))
    sigmas = np.zeros((state['N'], H))
    infos = []
    for d in range(state['N']):
        p, s, info = fit_predict(state['hist'][d], state['obs'][d], state['open_d'], state['promo_d'], state['school_d'], state['state_d'], state['woy'], L, H)
        preds[d] = p
        sigmas[d] = s
        infos.append(info)
    state['struct'] = preds
    state['sigma'] = sigmas
    state['info'] = infos
    return (prepared, state)
COV_OBS_PER_SERIES = 0.0

def select_context(variables, card, state):
    C = int(min(variables['cutoff_index'], card['limits']['max_context']))
    H = variables['horizon']
    po = variables.get('past_features')
    pf = variables.get('known_features')
    po = _as2d(po) if po is not None and len(po) else None
    pf = _as2d(pf) if pf is not None and len(pf) else None
    n_cov = (0 if po is None else po.shape[0]) + (0 if pf is None else pf.shape[0])
    use_cov = C >= COV_OBS_PER_SERIES * (state['N'] + n_cov)
    state['used_covariates'] = bool(use_cov)
    return ([{'target_indices': list(range(state['N'])), 'targets': _as2d(variables['target_history'])[:, -C:], 'past_only': po[:, -C:] if po is not None and use_cov else None, 'known_future': pf[:, -(C + H):] if pf is not None and use_cov else None, 'past_names': list(variables['past_names']) if use_cov else [], 'known_names': list(variables['known_names']) if use_cov else [], 'provenance': ('Full covariate context' if use_cov else 'Univariate context: too few observations per covariate') + '; structural specialist applied in postprocess'}], state)

def _theilsen(x, y):
    n = len(x)
    if n < 2:
        return 0.0
    sl = []
    for a in range(n - 1):
        dx = x[a + 1:] - x[a]
        ok = np.abs(dx) > 1e-09
        if ok.any():
            sl.append((y[a + 1:][ok] - y[a]) / dx[ok])
    return float(np.median(np.concatenate(sl))) if sl else 0.0

def postprocess(outputs, cases, state):
    N, H = (state['N'], state['H'])
    nat_p = np.empty((N, H))
    nat_q = np.empty((N, H, 9))
    for output, case in zip(outputs, cases):
        nat_p[case['target_indices']] = np.asarray(output['point'], float)
        nat_q[case['target_indices']] = np.asarray(output['quantiles'], float)
    nat_q = np.sort(nat_q, axis=2)
    L = state['L']
    corr_w = CORR_W_BASE + CORR_W_GAIN * L / (L + CORR_W_HALF)
    slope_damp = L / (L + SLOPE_HALF)
    shape_w = SHAPE_MAX * L / (L + SHAPE_HALF)
    closed = state['open_d'][L:] <= 1e-06
    exposure = state.get('exposure_future')
    typ = state.get('typ_open', 7.0)
    point = np.empty((N, H))
    quant = np.empty((N, H, 9))
    hh = np.arange(1, H + 1, dtype=float)
    for d in range(N):
        struct = state['struct'][d]
        sig = state['sigma'][d]
        nm = nat_q[d, :, 4]
        live = struct > 0
        if not live.any():
            med = np.zeros(H)
            rel = sig
            wn = 0.0
        else:
            lr = np.zeros(H)
            lr[live] = np.log(np.maximum(np.abs(nm[live]), 1e-09) / struct[live])
            dev = float(np.sqrt(np.mean(lr[live] ** 2))) / max(float(sig[0]), 1e-06)
            wn = corr_w / (1.0 + (max(dev - GUARD_TAU, 0.0) / GUARD_TAU) ** 2)
            normal = live & (exposure >= 0.9 * typ) if exposure is not None else live
            if normal.sum() < 2:
                normal = live
            lrn = lr[normal]
            hn = hh[normal]
            lvl = float(np.median(lrn))
            slope = _theilsen(hn, lrn) * slope_damp
            corr = lvl + slope * (hh - float(hn.mean()))
            corr = np.clip(corr, lrn.min() - 0.1, lrn.max() + 0.1)
            shape = np.clip(lr - corr, -3.0 * sig, 3.0 * sig)
            total = np.clip(wn * corr + shape_w * wn * 2.0 * shape, -3.0 * sig, 3.0 * sig)
            med = struct * np.exp(total)
            med = np.where(live, med, 0.0)
            nat_rel = (nat_q[d, :, 8] - nat_q[d, :, 0]) / (2.5631 * np.maximum(np.abs(nm), 1e-06))
            nat_rel = np.clip(nat_rel, 0.3 * sig, 3.0 * sig)
            ws = shape_w * wn / max(corr_w, 1e-09)
            rel = np.exp((1.0 - ws) * np.log(sig) + ws * np.log(np.maximum(nat_rel, 1e-06)))
            rel = np.clip(rel, 0.7 * sig, 1.5 * sig)
        if state['nonneg']:
            med = np.maximum(med, 0.0)
        safe = np.maximum(med, 1e-06)
        q = safe[:, None] * np.exp(rel[:, None] * (ZQ * ASYM)[None, :])
        if not state['nonneg']:
            q = med[:, None] + (rel * np.maximum(np.abs(med), 1e-06))[:, None] * ZQ[None, :]
        q = np.where(closed[:, None], 0.0, q)
        q = np.sort(q, axis=1)
        quant[d] = q
        point[d] = q[:, 4]
    return {'point': point, 'quantiles': quant}

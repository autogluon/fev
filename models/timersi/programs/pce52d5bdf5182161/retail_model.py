import numpy as np

def _weights(L, mask, halflife):
    age = (L - 1 - np.arange(L))[mask].astype(float)
    w = 0.5 ** (age / max(halflife, 1e-06))
    s = w.sum()
    return w / s * mask.sum() if s > 0 else np.ones(mask.sum())
GAMMA_HOLIDAY = 0.45

def effective_open(open_d, state_d, L, obs_mask=None, gamma=GAMMA_HOLIDAY):
    od = np.maximum(np.asarray(open_d, float), 0.0)
    past = od[:L]
    if obs_mask is not None and obs_mask.any():
        past = past[obs_mask]
    typ = float(np.percentile(past, 80)) if past.size else 7.0
    typ = max(typ, 1e-06)
    deficit = np.clip(typ - od, 0.0, None)
    holiday = np.minimum(np.maximum(np.asarray(state_d, float), 0.0), deficit)
    eff = od + (1.0 - gamma) * holiday
    eff = np.minimum(eff, typ)
    return (np.where(od <= 1e-06, 0.0, np.maximum(eff, 1e-06)), typ)

def _varies(x, thr=1e-06):
    return x.size > 0 and float(x.max() - x.min()) > thr

def build_features(open_d, promo_d, school_d, state_d, woy, L, H, n_eff):
    od = np.maximum(open_d, 0.0)
    f_promo = np.where(od > 0, promo_d / np.maximum(od, 1e-09), 0.0)
    f_school = np.where(od > 0, school_d / 7.0, 0.0)
    f_state = np.where(od > 0, state_d / 7.0, 0.0)
    t = np.arange(L + H, dtype=float)
    tnorm = (t - (L - 1)) / max(float(L), 1.0)
    cols = [np.ones(L + H)]
    names = ['const']
    pen = [1e-08]
    if n_eff >= 3 and _varies(f_promo[:L]):
        cols.append(f_promo)
        names.append('promo')
        pen.append(0.01)
    if n_eff >= 10 and _varies(f_school[:L]):
        cols.append(f_school)
        names.append('school')
        pen.append(0.05)
    if n_eff >= 16 and _varies(f_state[:L]):
        cols.append(f_state)
        names.append('state')
        pen.append(0.05)
    if n_eff >= 14:
        cols.append(tnorm)
        names.append('trend')
        pen.append(0.5)
    nharm = 2 if n_eff >= 78 else 1 if n_eff >= 48 else 0
    for k in range(nharm):
        cols.append(np.sin(2 * np.pi * (k + 1) * woy))
        names.append('sin%d' % k)
        pen.append(0.05)
        cols.append(np.cos(2 * np.pi * (k + 1) * woy))
        names.append('cos%d' % k)
        pen.append(0.05)
    return (np.column_stack(cols), names, np.asarray(pen, float))

def fit_predict(y, obs, open_d, promo_d, school_d, state_d, woy, L, H, halflife=52.0, shrink=0.25, prior_sigma=0.1, sigma_growth=0.02, gamma_holiday=GAMMA_HOLIDAY):
    od_raw = np.maximum(np.asarray(open_d, float), 0.0)
    y = np.asarray(y, float)
    obs = np.asarray(obs, bool)
    closed_future = od_raw[L:] <= 1e-06
    usable = obs & (od_raw[:L] > 0.3) & np.isfinite(y) & (y > 0)
    od, _typ = effective_open(od_raw, state_d, L, usable, gamma=gamma_holiday)
    n = int(usable.sum())
    if n == 0:
        base = float(np.nanmedian(y[obs])) if obs.any() and np.isfinite(y[obs]).any() else 0.0
        pred = np.where(closed_future, 0.0, max(base, 0.0))
        return (pred, np.full(H, prior_sigma), {'n': 0, 'names': [], 'coef': []})
    X, names, pen = build_features(od, promo_d, school_d, state_d, woy, L, H, n)
    W = _weights(L, usable, halflife)
    Xh_raw = X[:L][usable]
    centre = (Xh_raw * W[:, None]).sum(0) / n
    centre[0] = 0.0
    X = X - centre[None, :]
    Xh = X[:L][usable]
    u = np.log(y[usable] / od[:L][usable])
    p = Xh.shape[1]
    A = (Xh * W[:, None]).T @ Xh + np.diag(pen) * n
    b = np.linalg.solve(A + 1e-10 * np.eye(p), (Xh * W[:, None]).T @ u)
    if shrink > 0:
        b[1:] *= n / (n + shrink)
    resid = u - Xh @ b
    dof = max(float(n - p), 1.0)
    s2 = float(np.sum(W * resid ** 2) / dof) if n > p else prior_sigma ** 2
    k = dof / (dof + 4.0)
    sigma0 = float(np.sqrt(max(k * s2 + (1.0 - k) * prior_sigma ** 2, 1e-06)))
    sigma0 = float(np.clip(sigma0, 0.03, 0.45))
    Xf = X[L:].copy()
    lo = Xh.min(0)
    hi = Xh.max(0)
    span = hi - lo
    Xf = np.clip(Xf, lo - 0.25 * span, hi + 0.25 * span)
    fit_log = Xf @ b
    lo_lvl = np.log(np.maximum(y[usable] / od[:L][usable], 1e-09))
    fit_log = np.clip(fit_log, lo_lvl.min() - 0.8, lo_lvl.max() + 0.8)
    pred = od[L:] * np.exp(fit_log)
    pred = np.where(closed_future, 0.0, pred)
    hh = np.arange(1, H + 1, dtype=float)
    sigma = sigma0 * np.sqrt(1.0 + sigma_growth * hh)
    info = {'n': n, 'names': names, 'coef': b.tolist(), 'sigma0': sigma0}
    return (pred, sigma, info)

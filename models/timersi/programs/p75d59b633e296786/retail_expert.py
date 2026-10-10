import numpy as np
CAL_NAMES = ['xmas_ramp', 'dec_level', 'dec24', 'dec2730', 'dec31']
CAL_PRIOR = np.array([1.0, 0.0, -0.15, 0.1, -0.38])
ADVENT_BASE = 0.16
ADVENT_PEAK = 0.55
ADVENT_TAU = 7.0

def christmas_ramp(months, days):
    u = np.zeros(len(months), float)
    dd = np.where(months == 12, 24.0 - days, np.where(months == 11, 54.0 - days, 999.0))
    active = (dd >= 1.0) & (dd <= 30.0)
    u[active] = ADVENT_BASE + ADVENT_PEAK * np.exp(-(dd[active] - 1.0) / ADVENT_TAU)
    return u

def calendar_block(months, days):
    ramp = christmas_ramp(months, days)
    cols = [ramp, ((months == 12) & (days <= 23) | (months == 11) & (days >= 25)).astype(float), ((months == 12) & (days == 24)).astype(float), ((months == 12) & (days >= 27) & (days <= 30)).astype(float), ((months == 12) & (days == 31)).astype(float)]
    return np.stack(cols, axis=1)

def closure_proximity(openv, dow):
    n = len(openv)
    closed = openv < 0.5
    nxt = np.zeros(n, bool)
    nxt[:-1] = closed[1:]
    nxt2 = np.zeros(n, bool)
    nxt2[:-2] = closed[2:]
    prv = np.zeros(n, bool)
    prv[1:] = closed[:-1]
    prv2 = np.zeros(n, bool)
    prv2[2:] = closed[:-2]
    open_now = ~closed
    pre = (open_now & nxt & (dow <= 5)).astype(float)
    post = (open_now & prv & (dow >= 2)).astype(float)
    bridge = (open_now & nxt & nxt2).astype(float)
    after2 = (open_now & prv & prv2).astype(float)
    return (pre, post, bridge, after2)

def design_matrix(dyn, months, days, dom_end, L, sth_modal):
    n = len(months)
    dow = dyn['DayOfWeek']
    promo = dyn['Promo']
    cols, names = ([], [])
    for k in range(1, 8):
        cols.append((dow == k).astype(float))
        names.append('dow%d' % k)
    cols.append(promo.astype(float))
    names.append('promo')
    for k in range(1, 7):
        cols.append(((dow == k) & (promo == 1)).astype(float))
        names.append('pdow%d' % k)
    cols.append(dyn['SchoolHoliday'].astype(float))
    names.append('school')
    cols.append((dyn['StateHoliday'] != sth_modal).astype(float))
    names.append('stateh')
    if HOLIDAY_TYPE_SPLIT:
        sth = dyn['StateHoliday']
        open_tr = (np.arange(n) < L) & (dyn['Open'] > 0.5)
        codes = [c for c in np.unique(sth) if c != sth_modal]
        if len(codes) > 1:
            for c in codes:
                if ((sth == c) & open_tr).sum() >= 3:
                    cols.append((sth == c).astype(float))
                    names.append('sth_code_%g' % c)
    cols.append((days <= 3).astype(float))
    names.append('month_start')
    cols.append((days >= dom_end - 2).astype(float))
    names.append('month_end')
    if MONTH_HARMONICS:
        phase = 2.0 * np.pi * (days - 1) / np.maximum(dom_end, 1)
        for hmc in range(1, MONTH_HARMONICS + 1):
            cols.append(np.sin(hmc * phase))
            names.append('dom_sin%d' % hmc)
            cols.append(np.cos(hmc * phase))
            names.append('dom_cos%d' % hmc)
    pre, post, bridge, after2 = closure_proximity(dyn['Open'], dow)
    governed = ((months == 12) & (days >= 24)).astype(float)
    pre = pre * (1 - governed)
    post = post * (1 - governed)
    bridge = bridge * (1 - governed)
    after2 = after2 * (1 - governed)
    cols.append(pre)
    names.append('pre_closed')
    cols.append(post)
    names.append('post_closed')
    cols.append(bridge)
    names.append('pre_long_closure')
    cols.append(after2)
    names.append('post_long_closure')
    cols.append((np.arange(n) - (L - 1)) / 365.0)
    names.append('trend')
    cal = calendar_block(months, days)
    prior = np.zeros(len(names) + len(CAL_NAMES))
    for j, nm in enumerate(CAL_NAMES):
        cols.append(cal[:, j])
        names.append(nm)
        prior[len(names) - 1] = CAL_PRIOR[j]
    return (np.stack(cols, axis=1), names, prior, cal)
HALFLIFE = 130.0
LAM = 0.15
CAL_STRENGTH = 4.0
TREND_DAMP = 0.4
TREND_TAU = 0.0
ROBUST_ITERS = 2
ROBUST_C = 1.75
MONTH_HARMONICS = 2
HOLIDAY_TYPE_SPLIT = True

def fit_item(dyn, months, days, dom_end, y_hist, L, observed=None, halflife=None, lam=None, cal_strength=None, trend_damp=None):
    halflife = HALFLIFE if halflife is None else halflife
    lam = LAM if lam is None else lam
    cal_strength = CAL_STRENGTH if cal_strength is None else cal_strength
    trend_damp = TREND_DAMP if trend_damp is None else trend_damp
    'Weighted ridge on log sales of open days, shrunk toward the calendar prior.'
    n = len(months)
    vals, counts = np.unique(dyn['StateHoliday'][:L], return_counts=True)
    sth_modal = vals[np.argmax(counts)]
    X, names, prior, cal = design_matrix(dyn, months, days, dom_end, L, sth_modal)
    idx = np.arange(n)
    y_full = np.concatenate([y_hist, np.full(n - L, np.nan)])
    finite = np.isfinite(y_full)
    if observed is not None:
        finite[:L] &= np.asarray(observed, bool)
    train = (idx < L) & (dyn['Open'] > 0.5) & finite & (np.nan_to_num(y_full) > 0)
    out = {'names': names, 'cal': cal, 'ok': False, 'train': train}
    if train.sum() < 30:
        pos = y_hist[np.isfinite(y_hist) & (np.nan_to_num(y_hist) > 0)]
        out['beta'] = np.zeros(X.shape[1])
        out['mu'] = np.log(max(pos.mean() if pos.size else 1.0, 1.0))
        out['sigma'] = 0.3
        out['fit'] = np.full(n, np.exp(out['mu']))
        out['cal_series'] = np.zeros(n)
        return out
    Xtr = X[train]
    ytr = np.log(y_full[train])
    age = (L - 1 - idx[train]).astype(float)
    w = 0.5 ** (age / halflife)
    mu = np.average(ytr, weights=w)
    pen = np.full(X.shape[1], lam) * (w.sum() / 50.0)
    for nm in CAL_NAMES:
        pen[names.index(nm)] = cal_strength
    P = np.diag(pen)
    wr = w.copy()
    beta = None
    for _ in range(ROBUST_ITERS + 1):
        mu = np.average(ytr, weights=wr)
        A = Xtr * np.sqrt(wr)[:, None]
        b = (ytr - mu) * np.sqrt(wr)
        try:
            beta = np.linalg.solve(A.T @ A + P, A.T @ b + P @ prior)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(A.T @ A + P, A.T @ b + P @ prior, rcond=None)[0]
        if ROBUST_ITERS <= 0:
            break
        r = ytr - (mu + Xtr @ beta)
        scale = np.sqrt(max(np.average(r ** 2, weights=wr), 1e-08))
        wr = w * np.minimum(1.0, ROBUST_C * scale / np.maximum(np.abs(r), 1e-08))
    w = wr
    beta[names.index('trend')] *= trend_damp
    if TREND_TAU > 0:
        tcol = names.index('trend')
        t_days = (idx - (L - 1)).astype(float)
        fut = t_days > 0
        X = X.copy()
        X[fut, tcol] = TREND_TAU * (1.0 - np.exp(-t_days[fut] / TREND_TAU)) / 365.0
    resid = ytr - (mu + Xtr @ beta)
    sigma = float(np.sqrt(max(np.average(resid ** 2, weights=w), 1e-06)))
    order = np.argsort(resid)
    rs = resid[order]
    ws = w[order]
    cw = (np.cumsum(ws) - 0.5 * ws) / ws.sum()
    eq = np.interp(np.arange(1, 10) / 10.0, cw, rs)
    cal_support = float(w[np.abs(cal[:L][train[:L]]).sum(axis=1) > 0].sum()) if train.sum() else 0.0
    seasonal_idx = [j for j, nm in enumerate(names) if nm != 'trend']
    out.update(ok=True, beta=beta, mu=mu, sigma=sigma, emp_q=eq, cal_support=cal_support, fit=np.exp(mu + X @ beta), seasonal=np.exp(X[:, seasonal_idx] @ beta[seasonal_idx]), cal_series=cal @ beta[[names.index(nm) for nm in CAL_NAMES]])
    return out
WINSOR_Z = 2.8
WINSOR_WIN = 29

def deseasonalise(y_hist, seasonal, usable, L):
    s = np.maximum(seasonal[:L], 1e-06)
    z = np.where(usable[:L], np.maximum(np.nan_to_num(y_hist), 1.0) / s, np.nan)
    idx = np.arange(L)
    good = np.isfinite(z)
    if good.sum() < 10:
        return None
    z = np.interp(idx, idx[good], z[good])
    if not np.isfinite(z).all() or z.min() <= 0:
        return None
    if WINSOR_Z > 0 and L >= WINSOR_WIN:
        lz = np.log(z)
        half = WINSOR_WIN // 2
        med = np.empty(L)
        for t in range(L):
            lo, hi = (max(0, t - half), min(L, t + half + 1))
            med[t] = np.median(lz[lo:hi])
        mad = np.median(np.abs(lz - med)) * 1.4826
        if mad > 1e-06:
            lim = WINSOR_Z * mad
            lz = med + np.clip(lz - med, -lim, lim)
            z = np.exp(lz)
    return z

def annual_factor(level, L, H, lag=364, half_window=14, min_history=330):
    if level is None or L < min_history:
        return None

    def smooth(centre):
        lo = max(0, centre - half_window)
        hi = min(L, centre + half_window + 1)
        if hi - lo < half_window:
            return None
        return float(np.mean(level[lo:hi]))
    ref = smooth(L - 1 - lag)
    if not ref or ref <= 0:
        return None
    out = np.ones(H)
    for h in range(H):
        v = smooth(L + h - lag)
        if v is None or v <= 0:
            out[h] = out[h - 1] if h else 1.0
        else:
            out[h] = v / ref
    return out

def calendar_reference(cal_series, L, window=56, decay=28.0):
    n = min(window, L)
    if n <= 0:
        return 0.0
    w = 0.5 ** (np.arange(n)[::-1] / decay)
    return float(np.average(cal_series[L - n:L], weights=w))

import numpy as np
ZQ = np.array([-1.2816, -0.8416, -0.5244, -0.2533, 0.0, 0.2533, 0.5244, 0.8416, 1.2816])
EPS = 1e-12
CFG = dict(lwin=7, nwin=3, shrink=0.8, bias=-0.01, gdn=-0.08, gup=0.012, phi=0.97, prior_a=0.22, prior_b=0.55, wprior=4.0, sigmul=0.9, sigcap=0.3)

def _observed(hist, obs):
    y = np.asarray(hist, float).copy()
    y[~np.asarray(obs, bool)] = np.nan
    return y

def _mean(seg):
    seg = seg[np.isfinite(seg)]
    return seg.mean() if seg.size else np.nan

def level_estimate(d, win=7):
    n = len(d)
    if n == 0:
        return (0.0, 0.0)
    w = min(win, n)
    m = _mean(d[n - w:])
    if not np.isfinite(m):
        m = _mean(d[max(0, n - 3 * w):])
    if not np.isfinite(m):
        m = 0.0
    return (float(m), (w - 1) / 2.0)

def gamma_ratio(d, win=7, nwin=3, decay=0.6):
    n = len(d)
    ms = []
    for j in range(nwin + 1):
        hi = n - win * j
        lo = hi - win
        if lo < 0:
            break
        ms.append(_mean(d[lo:hi]))
    lr, wt = ([], [])
    for j in range(len(ms) - 1):
        a, b = (ms[j], ms[j + 1])
        if np.isfinite(a) and np.isfinite(b) and (a > EPS) and (b > EPS):
            lr.append(np.log(a / b) / win)
            wt.append(decay ** j)
    if not lr:
        return 0.0
    return float(np.average(lr, weights=wt))

def project(d, H, cfg):
    lev, off = level_estimate(d, cfg['lwin'])
    g = gamma_ratio(d, nwin=cfg['nwin'])
    ge = float(np.clip(cfg['shrink'] * g + cfg['bias'], cfg['gdn'], cfg['gup']))
    phi = cfg['phi']
    h = np.arange(1, H + 1, dtype=float)
    A = phi * (1 - phi ** h) / (1 - phi) if phi < 1 else h
    lev = max(lev, 0.0)
    return (lev * np.exp(ge * (A + off * phi)), lev, ge)

def backtest_resid(y, H, cfg):
    L = len(y)
    res = {}
    origins = [o for o in range(L - 7, 20, -7)][:6]
    for oi, o in enumerate(origins):
        if o < 21 or o >= L:
            continue
        past = y[:o]
        d = np.diff(past)
        if np.isfinite(d).sum() < 14:
            continue
        hmax = min(H, L - o)
        g, lev, _ = project(d, hmax, cfg)
        Sp = np.cumsum(g)
        fin = past[np.isfinite(past)]
        if not fin.size:
            continue
        anchor = fin[-1]
        Sa = y[o:o + hmax] - anchor
        c = max(0.5 * lev, 1.0)
        w = 0.8 ** oi
        for h in range(hmax):
            if not np.isfinite(Sa[h]):
                continue
            res.setdefault(h + 1, []).append((np.log((max(Sa[h], 0.0) + c) / (max(Sp[h], 0.0) + c)), w))
    return res

def sigma_profile(res_all, H, cfg):
    hs, vs, ws = ([], [], [])
    for h, lst in res_all.items():
        if not lst:
            continue
        r = np.array([x[0] for x in lst])
        w = np.array([x[1] for x in lst])
        vs.append(max(float(np.sqrt(np.average(r * r, weights=w))), 0.001))
        hs.append(h)
        ws.append(w.sum())
    h = np.arange(1, H + 1, dtype=float)
    prior = cfg['prior_a'] * h ** cfg['prior_b']
    if len(hs) < 6:
        return prior
    hs = np.array(hs, float)
    vs = np.array(vs)
    ws = np.array(ws, float)
    X, Y = (np.log(hs), np.log(vs))
    xb = np.average(X, weights=ws)
    yb = np.average(Y, weights=ws)
    den = float(np.sum(ws * (X - xb) ** 2))
    b = float(np.sum(ws * (X - xb) * (Y - yb)) / den) if den > EPS else cfg['prior_b']
    b = float(np.clip(b, 0.2, 1.0))
    a = float(np.clip(np.exp(yb - b * xb), 0.05, 1.2))
    fit = a * h ** b
    n = float(ws.sum())
    lam = n / (n + cfg['wprior'] * H)
    return np.exp(lam * np.log(fit) + (1 - lam) * np.log(prior))

def forecast_item(th, ob, H, cfg):
    D = th.shape[0]
    res_all, series = ({}, [])
    for d in range(D):
        y = _observed(th[d], ob[d])
        fin = y[np.isfinite(y)]
        anchor = float(fin[-1]) if fin.size else 0.0
        g, lev, ge = project(np.diff(y), H, cfg)
        series.append((anchor, g, lev, ge))
        for h, lst in backtest_resid(y, H, cfg).items():
            res_all.setdefault(h, []).extend(lst)
    sig = sigma_profile(res_all, H, cfg)
    sig = np.clip(sig * cfg['sigmul'], 0.03, cfg['sigcap'])
    point = np.zeros((D, H))
    quant = np.zeros((D, H, 9))
    for d in range(D):
        anchor, g, lev, ge = series[d]
        S = np.cumsum(g)
        point[d] = anchor + S
        c = max(0.5 * lev, 1.0)
        for j, z in enumerate(ZQ):
            quant[d, :, j] = anchor + np.maximum((S + c) * np.exp(sig * z) - c, 0.0)
        quant[d] = np.sort(quant[d], axis=1)
        quant[d] = np.maximum(quant[d], anchor)
        point[d] = np.clip(point[d], quant[d, :, 0], quant[d, :, 8])
    return (point, quant, sig)

def preprocess(view, card):
    th = np.asarray(view['target_history'], float)
    ob = np.asarray(view['target_observed'], bool)
    state = {'N': th.shape[0], 'H': int(view['horizon']), 'th': th, 'ob': ob, 'item': view.get('item_id')}
    return (view, state)

def engineer(prepared, card, state):
    return (prepared, state)

def select_context(variables, card, state):
    C = min(variables['cutoff_index'], card['limits']['max_context'])
    H = variables['horizon']
    po = variables['past_features']
    pf = variables['known_features']
    return ([{'target_indices': list(range(state['N'])), 'targets': variables['target_history'][:, -C:], 'past_only': po[:, -C:] if len(po) else None, 'known_future': pf[:, -(C + H):] if len(pf) else None, 'past_names': variables['past_names'], 'known_names': variables['known_names'], 'provenance': 'Reference native inputs; forecast rebuilt in increment space'}], state)

def postprocess(outputs, cases, state):
    N, H = (state['N'], state['H'])
    native = np.empty((N, H))
    for output, case in zip(outputs, cases):
        native[case['target_indices']] = output['point']
    point, quant, sig = forecast_item(state['th'], state['ob'], H, CFG)
    return {'point': point, 'quantiles': quant, 'components': {'native_point': native, 'sigma': sig}}

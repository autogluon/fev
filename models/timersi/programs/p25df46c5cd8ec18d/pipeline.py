import numpy as np
_PANEL = {}
G_TREND = 0.35
MU_POOL = 0.2
KAPPA = 0.6
WIDEN = 1.25
TAU_SKEW = 0.0
_Z = np.array([-1.2815515655446004, -0.8416212335729143, -0.5244005127080407, -0.2533471031357997, 0.0, 0.2533471031357997, 0.5244005127080407, 0.8416212335729143, 1.2815515655446004])

def _theil_sen(t, y):
    n = len(y)
    if n < 2:
        return (0.0, float(y[0]) if n else 0.0, np.zeros(n))
    if n == 2:
        sl = (y[1] - y[0]) / max(t[1] - t[0], 1e-09)
    else:
        i, j = np.triu_indices(n, 1)
        dt = t[j] - t[i]
        ok = dt > 0
        sl = float(np.median((y[j][ok] - y[i][ok]) / dt[ok])) if ok.any() else 0.0
    a = float(np.median(y - sl * t))
    return (float(sl), a, y - (a + sl * t))

def _slope_change(y, H):
    d = np.diff(y)
    if d.size < 2:
        return 0.0
    w = int(max(1, min(H, d.size // 2)))
    return float(d[-w:].mean() - d[-2 * w:-w].mean())

def _mad(v):
    v = np.asarray(v, float)
    if v.size == 0:
        return 0.0
    return float(1.4826 * np.median(np.abs(v - np.median(v))))

def preprocess(view, card):
    hist = np.asarray(view['target_history'], float)
    obs = np.asarray(view.get('target_observed'), bool) if len(view.get('target_observed', [])) else np.isfinite(hist)
    D, L = hist.shape
    H = int(view['horizon'])
    stats = []
    win = int(min(L, max(2 * H, 8)))
    for d in range(D):
        y = hist[d]
        m = obs[d] & np.isfinite(y)
        idx = np.flatnonzero(m)
        if idx.size == 0:
            stats.append(dict(ok=False))
            continue
        yv = y[idx].astype(float)
        keep = idx >= L - win
        tw, yw = (idx[keep].astype(float), yv[keep])
        if tw.size < 2:
            tw, yw = (idx.astype(float), yv)
        sl, a, res = _theil_sen(tw, yw)
        level_fit = a + sl * (L - 1)
        st = dict(ok=True, sl=sl, last=float(yv[-1]), anchor=float(level_fit), resid=max(_mad(res), 0.0), dsl=_slope_change(yv, H), n=int(idx.size))
        stats.append(st)
        if st['n'] >= 3:
            _PANEL.setdefault(int(view['cutoff_index']), {})[view.get('item_id'), d] = (st['sl'], st['dsl'])
    return (view, {'N': D, 'H': H, 'L': L, 'stats': stats, 'cut': int(view['cutoff_index'])})

def engineer(prepared, card, state):
    L, H = (state['L'], state['H'])
    t = np.arange(L + H, dtype=float) - (L - 1)
    good = [s for s in state['stats'] if s.get('ok')]
    chans, names = ([], [])
    if good:
        sl = float(np.median([s['sl'] for s in good]))
        chans.append(sl * t)
        names.append('causal_trend_est')
    pan = _panel_stats(state)
    if pan is not None:
        chans.append(pan[0] * t)
        names.append('panel_factor_est')
    state['extra_known'] = np.asarray(chans, float) if chans else None
    state['extra_known_names'] = names
    state['panel_at_engineer'] = 0 if pan is None else pan[2]
    return (prepared, state)

def select_context(variables, card, state):
    C = min(variables['cutoff_index'], card['limits']['max_context'])
    H = variables['horizon']
    po = variables['past_features']
    pf = variables['known_features']
    kf = pf[:, -(C + H):] if len(pf) else None
    knames = list(variables['known_names'])
    ek = state.get('extra_known')
    if ek is not None and ek.ndim == 2 and (ek.shape[1] >= C + H) and np.all(np.isfinite(ek)):
        ek = ek[:, -(C + H):]
        kf = ek if kf is None else np.concatenate([kf, ek], axis=0)
        knames = knames + state['extra_known_names']
    return ([{'target_indices': list(range(state['N'])), 'targets': variables['target_history'][:, -C:], 'past_only': po[:, -C:] if len(po) else None, 'known_future': kf, 'past_names': variables['past_names'], 'known_names': knames, 'provenance': 'Reference targets + causal trend / panel common-factor known-future estimates (panel members at engineer: %d)' % state['panel_at_engineer']}], state)

def _panel_stats(state):
    reg = _PANEL.get(state['cut'], {})
    if len(reg) < 8:
        return None
    sls = np.array([v[0] for v in reg.values()], float)
    dsl = np.array([v[1] for v in reg.values()], float)
    return (float(np.median(sls)), max(_mad(dsl), 0.0), len(reg))

def postprocess(outputs, cases, state):
    N, H = (state['N'], state['H'])
    point = np.zeros((N, H))
    quant = np.zeros((N, H, 9))
    for output, case in zip(outputs, cases):
        point[case['target_indices']] = np.asarray(output['point'], float)
        quant[case['target_indices']] = np.asarray(output['quantiles'], float)
    nat_p, nat_q = (point.copy(), quant.copy())
    panel = _panel_stats(state)
    h = np.arange(1, H + 1, dtype=float)
    for d in range(N):
        st = state['stats'][d]
        if not st.get('ok') or not np.all(np.isfinite(nat_p[d])) or (not np.all(np.isfinite(nat_q[d]))):
            continue
        sl = st['sl']
        if panel is not None:
            pool_sl, sig_pool, _ = panel
            sl_adj = sl + MU_POOL * (pool_sl - sl)
        else:
            sig_pool = 0.0
            sl_adj = sl
        base = 0.5 * st['last'] + 0.5 * st['anchor']
        lin = base + sl_adj * h
        p = (1.0 - G_TREND) * nat_p[d] + G_TREND * lin
        dev_nat = (nat_q[d] - nat_p[d][:, None]) * WIDEN
        sig_s = max(abs(st['dsl']), sig_pool)
        sig = KAPPA * np.sqrt(st['resid'] ** 2 + (sig_s * h) ** 2)
        dev_floor = _Z[None, :] * sig[:, None]
        if TAU_SKEW and panel is not None:
            sk = TAU_SKEW * (panel[0] - sl_adj) * h
            dev_floor = dev_floor + np.where(np.sign(_Z)[None, :] * np.sign(sk)[:, None] > 0, sk[:, None], 0.0)
        dev = np.where(np.abs(dev_floor) > np.abs(dev_nat), dev_floor, dev_nat)
        point[d] = p
        quant[d] = np.sort(p[:, None] + dev, axis=1)
    return {'point': point, 'quantiles': quant}

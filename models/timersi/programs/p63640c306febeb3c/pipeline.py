import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
LEVELS = np.arange(1, 10) / 10.0
BETAS = (0.0, 0.7, 1.0)
GAMMAS = (0.0, 0.5, 1.0)
SPREADS = (0.75, 0.85, 1.0)
SPREADS_LO = (0.7, 0.85, 1.0)
SPREAD_PASS = 1.002
QUALITY_FLOOR = 0.0
RIDGE_ALPHA = 5.0
RHAT_CLIP = 0.3
WEATHER_COLS = ('AH', 'RH', 'T')
DEFAULT_BETA = 0.7
HALF_LIFE = 28.0
BIAS_LAMBDA = 0.5
BIAS_CLIP = 0.18
BIAS_RECENT_RULE = True
BIAS_LEAD_HL = 0.0
BIAS_VAC_GUARD = True
HOL_SHRINK = 0.8
MIN_FIT_DAYS = 21
HOLIDAYS = {'2004-04-12', '2004-04-25', '2004-05-01', '2004-06-02', '2004-08-15', '2004-11-01', '2004-12-08', '2004-12-25', '2004-12-26', '2005-01-01', '2005-01-06', '2005-03-28', '2005-04-25', '2005-05-01'}
VLAMS = (0.0, 0.7, 1.0)
DEFAULT_VLAM = 0.7
SUN_KAPPA = 0.9
WIN_KAPPA = 0.35
WIN_KAPPA_BY_TARGET = None
VAC_CLIP = 0.8
VAC_ANCHOR_HL = 7.0
VAC_EXIT_FLOOR = True
VAC_EXIT_THR = 0.35
VAC_FLOOR_SHRINK = 1.0
VAC_WIDEN = 2.0

def _vac_calendar(ts):
    tt = pd.to_datetime(list(ts))
    v = np.zeros(len(tt))
    typ = np.zeros(len(tt), int)
    for j, t in enumerate(tt):
        m, dd = (t.month, t.day)
        if m == 8:
            typ[j] = 1
            if dd <= 7:
                v[j] = 0.4
            elif dd <= 14:
                v[j] = 0.7
            elif dd <= 29:
                v[j] = 1.0
            else:
                v[j] = 0.3 if dd == 30 else 0.0
        elif m == 12 and dd >= 24 or (m == 1 and dd <= 6):
            typ[j] = 2
            v[j] = 1.0
    return (v, typ)

def _vac_deltas(hist, v, typ, dows, hol, profs):
    N = hist.shape[0]
    d6 = np.where(hol, 6, dows)
    ds = np.full(N, np.nan)
    dw = np.full(N, np.nan)
    for i in range(N):
        ly = np.log(np.maximum(hist[i], 1e-06))
        adj = ly - profs[i][d6]
        for tcode, out in ((1, ds), (2, dw)):
            core = (typ == tcode) & (v >= 0.7) & np.isfinite(adj)
            if core.sum() < 5:
                continue
            idx = np.where(core)[0]
            near = np.zeros(len(v), bool)
            near[max(0, idx.min() - 45):idx.max() + 46] = True
            base = (v == 0) & np.isfinite(adj) & near
            if base.sum() >= 7:
                out[i] = adj[core].mean() - adj[base].mean()
    sun = profs[:, 6] - profs[:, :5].mean(axis=1)
    ds = np.where(np.isfinite(ds), ds, SUN_KAPPA * sun)
    wk = WIN_KAPPA_BY_TARGET if WIN_KAPPA_BY_TARGET is not None else np.full(N, WIN_KAPPA)
    dw = np.where(np.isfinite(dw), dw, wk[:N] * ds)
    ds = np.clip(ds, -0.9, 0.15)
    dw = np.clip(dw, -0.9, 0.15)
    return (ds, dw)

def _vac_corr(hist, v_past, typ_past, v_fut, typ_fut, dows, hol, profs, lam):
    ds, dw = _vac_deltas(hist, v_past, typ_past, dows, hol, profs)
    N = hist.shape[0]
    H = len(v_fut)
    corr = np.zeros((N, H))
    n = len(v_past)
    k = min(n, 28)
    w = 0.5 ** ((k - 1 - np.arange(k)) / VAC_ANCHOR_HL)
    for i in range(N):
        e_past = np.where(typ_past == 1, ds[i] * v_past, np.where(typ_past == 2, dw[i] * v_past, 0.0))
        e_fut = np.where(typ_fut == 1, ds[i] * v_fut, np.where(typ_fut == 2, dw[i] * v_fut, 0.0))
        anchor = np.sum(w * e_past[-k:]) / max(np.sum(w), 1e-09)
        corr[i] = np.clip(lam[i] * (e_fut - anchor), -VAC_CLIP, VAC_CLIP)
    return (corr, ds, dw)

def _vac_exit_floor(hist, v_past, typ_past, v_fut, dows, hol, profs, anchor_v):
    N, n = hist.shape
    d6 = np.where(hol, 6, dows)
    lvl = np.full(N, np.nan)
    sel = v_past == 0
    if sel.sum() < 14:
        return lvl
    w = 0.5 ** ((n - 1 - np.arange(n)) / 28.0) * sel
    for i in range(N):
        ly = np.log(np.maximum(hist[i], 1e-06))
        adj = ly - profs[i][d6]
        fin = np.isfinite(adj) & sel
        if fin.sum() < 14:
            continue
        lvl[i] = np.sum(w[fin] * adj[fin]) / np.sum(w[fin])
    return lvl

def _vac_widen(q, corr):
    med = q[..., 4:5]
    d = q - med
    hi = (1.0 + VAC_WIDEN * np.maximum(corr, 0.0))[..., None]
    lo = (1.0 + VAC_WIDEN * np.maximum(-corr, 0.0))[..., None]
    return np.sort(med + np.where(d > 0, d * hi, d * lo), axis=-1)

def _dow(ts):
    return pd.to_datetime(list(ts)).dayofweek.values

def _is_hol(ts):
    return np.array([str(t)[:10] in HOLIDAYS for t in pd.to_datetime(list(ts))])

def _twoway(y, dows, w=None, hol=None):
    ly = np.log(np.maximum(np.asarray(y, float), 1e-06))
    n = len(ly)
    if w is None:
        w = np.ones(n)
    d = dows.copy()
    if hol is not None:
        d = np.where(hol, 6, d)
    fin = np.isfinite(ly)
    wk = np.arange(n) // 7
    nw = wk.max() + 1
    p = np.zeros(7)
    for _ in range(8):
        lvl = np.zeros(nw)
        for k in range(nw):
            s = (wk == k) & fin
            lvl[k] = np.sum((ly[s] - p[d[s]]) * w[s]) / max(np.sum(w[s]), 1e-09) if s.sum() else np.nan
        r = ly - np.where(np.isfinite(lvl), lvl, np.nanmean(lvl))[wk]
        new = np.zeros(7)
        cnt = np.zeros(7)
        for dd in range(7):
            s = (d == dd) & fin & np.isfinite(r)
            if s.sum():
                new[dd] = np.sum(r[s] * w[s]) / max(np.sum(w[s]), 1e-09)
                cnt[dd] = np.sum(w[s])
        new -= new.mean()
        p = new
    p = p * (cnt / (cnt + 1.0))
    return p - p.mean()

def _profiles(hist, dows, hol):
    D, n = hist.shape
    w = 0.5 ** ((n - 1 - np.arange(n)) / HALF_LIFE)
    profs = np.array([_twoway(hist[i], dows, w, hol) for i in range(D)])
    pool = profs.mean(axis=0)
    out = 0.5 * profs + 0.5 * pool[None, :]
    return out - out.mean(axis=1, keepdims=True)

def _fcst_profile(pred, fdow):
    return _twoway(pred, fdow)

def _corr_factors(profs, pred, fdow, fhol, beta):
    fp = _fcst_profile(pred, fdow)
    resid = profs[fdow] - fp[fdow]
    corr = beta * resid
    if fhol.any():
        hcorr = HOL_SHRINK * (profs[6] - fp[fdow])
        corr = np.where(fhol & (fdow != 6), hcorr, corr)
    return np.exp(np.clip(corr, -0.7, 0.4))

def _weather_fit(hist, dows, hol, profs, Xp):
    if Xp is None or Xp.shape[0] < MIN_FIT_DAYS:
        return None
    ok_any = np.isfinite(Xp).all(axis=1)
    if ok_any.sum() < MIN_FIT_DAYS:
        return None
    mu, sd = (Xp[ok_any].mean(0), Xp[ok_any].std(0) + 1e-09)
    d6 = np.where(hol, 6, dows)
    n = hist.shape[1]
    wk = np.arange(n) // 7
    models = []
    for i in range(hist.shape[0]):
        ly = np.log(np.maximum(hist[i], 1e-06))
        lvl = np.array([np.nanmean(ly[wk == k] - profs[i][d6[wk == k]]) for k in range(wk.max() + 1)])
        r = ly - lvl[wk] - profs[i][d6]
        ok = np.isfinite(r) & ok_any
        if ok.sum() < MIN_FIT_DAYS:
            models.append(None)
            continue
        models.append(Ridge(alpha=RIDGE_ALPHA).fit((Xp[ok] - mu) / sd, r[ok]))

    def predict(Xf, anchor):
        Xf = np.where(np.isfinite(Xf), Xf, anchor[None, :])
        out = []
        for m in models:
            if m is None:
                out.append(np.zeros(Xf.shape[0]))
                continue
            rh = m.predict((Xf - mu) / sd) - m.predict(((anchor - mu) / sd)[None, :])[0]
            out.append(np.clip(rh, -RHAT_CLIP, RHAT_CLIP))
        return np.array(out)
    return predict
SPEC_ENABLE = False
SPEC_WS = (0.0, 0.25, 0.5)
SPEC_ALPHA = 3.0
SPEC_HL = 45.0
SPEC_MAXN = 200
SPEC_TREND_DAMP = 21.0

def _spec_points(hist, dows, hol, v_past, typ_past, v_fut, typ_fut, fdow, fhol, Xp, Xf, H):
    N, n = hist.shape
    out = np.full((N, H), np.nan)
    info = []
    d6 = np.where(hol, 6, dows)
    fd6 = np.where(fhol, 6, fdow)
    W = Wf = None
    if Xp is not None and Xp.shape[0] == n:
        W = np.stack([pd.Series(Xp[:, j]).interpolate(limit_direction='both').values for j in range(Xp.shape[1])], 1)
        anchor = np.nanmean(W[max(0, n - 14):], 0)
        Wf = np.where(np.isfinite(Xf), Xf, anchor[None, :]) if Xf is not None else np.tile(anchor, (H, 1))
        if not (np.isfinite(W).all() and np.isfinite(Wf).all()):
            W = Wf = None

    def feats(days, d6a, va, ta, wa):
        oh = np.zeros((len(days), 6))
        for dd in range(6):
            oh[:, dd] = d6a == dd
        cols = [oh, (va * (ta == 1))[:, None], (va * (ta == 2))[:, None]]
        if wa is not None:
            cols.append(wa)
        tt = (np.asarray(days, float) - (n - 1)) / SPEC_TREND_DAMP
        cols.append(np.tanh(tt)[:, None])
        return np.concatenate(cols, 1)
    t0 = max(0, n - SPEC_MAXN)
    idx = np.arange(t0, n)
    Xtr = feats(idx, d6[idx], v_past[idx], typ_past[idx], W[idx] if W is not None else None)
    Xte = feats(np.arange(n, n + H), fd6, v_fut, typ_fut, Wf)
    for i in range(N):
        ytr = np.log(np.maximum(hist[i, idx], 1e-06))
        ok = np.isfinite(ytr)
        if ok.sum() < 2 * MIN_FIT_DAYS:
            info.append(None)
            continue
        sw = 0.5 ** ((n - 1 - idx[ok]) / SPEC_HL)
        mu, sd = (Xtr[ok].mean(0), Xtr[ok].std(0) + 1e-09)
        m = Ridge(alpha=SPEC_ALPHA).fit((Xtr[ok] - mu) / sd, ytr[ok], sample_weight=sw)
        out[i] = m.predict((Xte - mu) / sd)
        info.append({'n_train': int(ok.sum()), 'coef_dow': np.round(m.coef_[:6], 3).tolist(), 'coef_vac': np.round(m.coef_[6:8], 3).tolist(), 'coef_weather': np.round(m.coef_[8:-1], 3).tolist() if W is not None else [], 'coef_trend': float(np.round(m.coef_[-1], 3))})
    return (out, info)

def _spread(q, s_lo, s_hi):
    med = q[..., 4:5]
    d = q - med
    return np.sort(med + np.where(d < 0, s_lo * d, s_hi * d), axis=-1)

def _impute(hist, dows, X_T):
    out = hist.copy()
    D, n = hist.shape
    oh = np.zeros((n, 6))
    for d in range(6):
        oh[:, d] = dows == d
    base = np.log(np.maximum(hist[0], 1e-06))
    b = pd.Series(base).interpolate(limit_direction='both').values
    out[0] = np.where(np.isfinite(hist[0]), hist[0], np.exp(b))
    cols = [b[:, None], oh]
    if X_T is not None:
        t = pd.Series(X_T).interpolate(limit_direction='both').values
        if np.isfinite(t).all():
            cols.insert(1, t[:, None])
    X = np.concatenate(cols, axis=1)
    for i in range(1, D):
        ly = np.log(np.maximum(hist[i], 1e-06))
        ok = np.isfinite(ly)
        if ok.sum() >= MIN_FIT_DAYS and (~ok).any():
            m = Ridge(alpha=2.0).fit(X[ok], ly[ok])
            pred = m.predict(X[~ok])
            out[i, ~ok] = np.exp(pred)
        s = pd.Series(np.log(np.maximum(out[i], 1e-06))).interpolate(limit_direction='both').values
        out[i] = np.where(np.isfinite(out[i]), out[i], np.exp(s))
    return out

def _pinball(truth, q):
    e = truth[..., None] - q
    return np.nanmean(2 * np.where(e >= 0, LEVELS * e, (LEVELS - 1) * e), axis=(-2, -1))

def preprocess(view, card):
    return (view, {'N': len(view['target_ids']), 'H': view['horizon'], 'L': view['cutoff_index']})

def engineer(prepared, card, state):
    ts = prepared['timestamps']
    L, H = (state['L'], state['H'])
    state['past_dow'] = _dow(ts[:L])
    state['past_hol'] = _is_hol(ts[:L])
    state['fut_dow'] = _dow(ts[L:L + H])
    state['fut_hol'] = _is_hol(ts[L:L + H])
    v_all, typ_all = _vac_calendar(ts[:L + H])
    state['vac_v'] = v_all
    state['vac_typ'] = typ_all
    global WIN_KAPPA_BY_TARGET
    WIN_KAPPA_BY_TARGET = np.array([0.0 if 'NO2' in str(n) else WIN_KAPPA for n in prepared['target_names']])
    hist_raw = np.asarray(prepared['target_history'], float)
    obs = np.asarray(prepared['target_observed'], bool)
    hist = np.where(obs, hist_raw, np.nan)
    state['hist'] = hist
    state['profiles'] = _profiles(hist, state['past_dow'], state['past_hol'])
    names0 = list(prepared['known_names'])
    tj = [j for j, n in enumerate(names0) if n == 'T']
    X_T = np.asarray(prepared['known_features'], float)[tj[0], :L] if tj else None
    state['hist_filled'] = _impute(hist, state['past_dow'], X_T)
    names = list(prepared['known_names'])
    widx = [j for j, n in enumerate(names) if n in WEATHER_COLS]
    state['widx'] = widx
    if widx:
        X = np.asarray(prepared['known_features'], float)[widx]
        state['Xp'] = X[:, :L].T
        state['Xf'] = X[:, L:L + H].T
        a = X[:, max(0, L - 14):L]
        state['anchor'] = np.nanmean(a, axis=1)
        state['weather'] = _weather_fit(hist, state['past_dow'], state['past_hol'], state['profiles'], state['Xp'])
    else:
        state['weather'] = None
    return (prepared, state)

def select_context(v, card, state):
    L, H, N = (state['L'], state['H'], state['N'])
    maxC = card['limits']['max_context']
    po = v['past_features']
    kf = v['known_features']
    ts = v['timestamps']
    hist = state['hist']
    filled = state['hist_filled']

    def case(offset, partial=False):
        end = L - offset
        C = min(end, maxC)
        row = {'target_indices': list(range(N)), 'targets': filled[:, end - C:end], 'past_only': np.asarray(po, float)[:, end - C:end] if len(po) else None, 'known_future': np.asarray(kf, float)[:, end - C:end + H] if len(kf) else None, 'past_names': v['past_names'], 'known_names': v['known_names'], 'provenance': {'arm': 'raw', 'offset': offset}}
        if offset:
            row['origin_offset'] = offset
            row['auxiliary'] = True
            if partial:
                row['partial_auxiliary'] = True
        return row
    cases = [case(0)]
    state['aux'] = []
    if L - H >= MIN_FIT_DAYS:
        offsets = [(14, True), (H, False)]
        if L - 2 * H >= MIN_FIT_DAYS:
            offsets.append((2 * H, False))
    else:
        offsets = [(off, True) for off in (7, 14) if L - off >= 10]
    for off, partial in offsets:
        cases.append(case(off, partial))
        m = min(off, H)
        end = L - off
        state['aux'].append({'offset': off, 'matured': m, 'end': end, 'truth': hist[:, end:end + m], 'dows': state['past_dow'][:end], 'hols': state['past_hol'][:end], 'fdow': _dow(ts[end:end + H]), 'fhol': _is_hol(ts[end:end + H]), 'v_past': state['vac_v'][:end], 'typ_past': state['vac_typ'][:end], 'v_fut': state['vac_v'][end:end + H], 'typ_fut': state['vac_typ'][end:end + H], 'weight': end / float(L), 'Xp': np.asarray(kf, float)[state['widx']][:, :end].T if state['widx'] else None, 'Xf': np.asarray(kf, float)[state['widx']][:, end:end + H].T if state['widx'] else None, 'anchor': np.nanmean(np.asarray(kf, float)[state['widx']][:, max(0, end - 14):end], axis=1) if state['widx'] else None})
    return (cases, state)

def postprocess(outputs, cases, state):
    N, H = (state['N'], state['H'])
    primary = outputs[0]
    beta_tbl, bias_tbl, wts, aux_meta = ([], [], [], [])
    vlam_tbl, vlam_info = ([], [])
    for j, aux in enumerate(state['aux']):
        out = outputs[1 + j]
        m = aux['matured']
        if aux['end'] < MIN_FIT_DAYS:
            continue
        profs = _profiles(state['hist'][:, :aux['end']], aux['dows'], aux['hols'])
        vac_by_lam = []
        for lv in VLAMS:
            vc, _, _ = _vac_corr(state['hist'][:, :aux['end']], aux['v_past'], aux['typ_past'], aux['v_fut'], aux['typ_fut'], aux['dows'], aux['hols'], profs, np.full(N, lv))
            vac_by_lam.append(vc)
        vac0, _, _ = _vac_corr(state['hist'][:, :aux['end']], aux['v_past'], aux['typ_past'], aux['v_fut'], aux['typ_fut'], aux['dows'], aux['hols'], profs, np.full(N, DEFAULT_VLAM))
        info = np.abs(vac_by_lam[-1][:, :m]).max(axis=1) > 0.03
        pb = np.zeros((N, len(BETAS)))
        pv = np.zeros((N, len(VLAMS)))
        bias = np.zeros(N)
        for i in range(N):
            vfac0 = np.exp(vac0[i])
            for bi, b in enumerate(BETAS):
                f = _corr_factors(profs[i], out['point'][i], aux['fdow'], aux['fhol'], b)
                qc = np.sort(out['quantiles'][i, :m] * (f * vfac0)[:m, None], axis=-1)
                pb[i, bi] = _pinball(aux['truth'][i], qc)
            fb = _corr_factors(profs[i], out['point'][i], aux['fdow'], aux['fhol'], DEFAULT_BETA)
            for li in range(len(VLAMS)):
                fl = fb * np.exp(vac_by_lam[li][i])
                ql = np.sort(out['quantiles'][i, :m] * fl[:m, None], axis=-1)
                pv[i, li] = _pinball(aux['truth'][i], ql)
            pc = out['point'][i, :m] * (fb * vfac0)[:m]
            bias[i] = np.nanmean(np.log(np.maximum(pc, 1e-06)) - np.log(np.maximum(aux['truth'][i], 1e-06)))
        wfit = _weather_fit(state['hist'][:, :aux['end']], aux['dows'], aux['hols'], profs, aux['Xp']) if aux['Xp'] is not None else None
        rhat = wfit(aux['Xf'], aux['anchor']) if wfit is not None else np.zeros((N, H))
        aux_meta.append({'out': out, 'm': m, 'truth': aux['truth'], 'profs': profs, 'rhat': rhat, 'fdow': aux['fdow'], 'fhol': aux['fhol'], 'vac_by_lam': vac_by_lam, 'aux': aux})
        beta_tbl.append(pb)
        bias_tbl.append(bias)
        wts.append(aux['weight'])
        vlam_tbl.append(pv)
        vlam_info.append(info)
    if beta_tbl:
        w = np.array(wts)[:, None, None]
        loss = np.array(beta_tbl)
        scale = np.maximum((loss[..., 0] * np.array(wts)[:, None]).sum(0) / sum(wts), 1e-09)
        obj = (loss / scale[None, :, None] * w).sum(0) / sum(wts)
        obj = obj + 0.01 * np.abs(np.array(BETAS) - DEFAULT_BETA)[None, :]
        beta = np.array(BETAS)[np.argmin(obj, axis=-1)]
    else:
        beta = np.full(N, DEFAULT_BETA)
    vlam = np.full(N, DEFAULT_VLAM)
    if vlam_tbl:
        VL = np.array(vlam_tbl)
        IN = np.array(vlam_info)
        WV = np.array(wts)
        for i in range(N):
            sel = IN[:, i]
            if sel.sum() == 0:
                continue
            lo = VL[sel, i, :]
            wv = WV[sel]
            scale = np.maximum((lo[:, VLAMS.index(DEFAULT_VLAM)] * wv).sum() / wv.sum(), 1e-09)
            obj = (lo * wv[:, None]).sum(0) / wv.sum() / scale
            obj = obj + 0.01 * np.abs(np.array(VLAMS) - DEFAULT_VLAM)
            vlam[i] = VLAMS[int(np.argmin(obj))]
    bias_hat = np.zeros(N)
    if bias_tbl:
        B = np.array(bias_tbl)
        W = np.array(wts)
        offs = np.array([a['offset'] for a in state['aux']][:B.shape[0]], float)
        if BIAS_RECENT_RULE and (offs <= 28).sum() >= 2:
            near = offs <= 28
            Bn, Wn = (B[near], W[near])
            same_sign = np.all(Bn > 0, axis=0) | np.all(Bn < 0, axis=0)
            wmean = (Bn * Wn[:, None]).sum(0) / Wn.sum()
            stale = ~near
            if stale.any():
                agree_all = same_sign & (np.sign(B[stale].sum(0)) == np.sign(wmean))
            else:
                agree_all = same_sign
            cap = np.where(agree_all, BIAS_CLIP, 0.5 * BIAS_CLIP)
            bias_hat = np.where(same_sign, np.clip(BIAS_LAMBDA * wmean, -cap, cap), 0.0)
        elif B.shape[0] >= 2:
            same_sign = np.all(B > 0, axis=0) | np.all(B < 0, axis=0)
            lam = BIAS_LAMBDA if B.shape[0] == 2 else 0.7
            wmean = (B * W[:, None]).sum(0) / W.sum()
            bias_hat = np.where(same_sign, np.clip(lam * wmean, -BIAS_CLIP, BIAS_CLIP), 0.0)
        else:
            bias_hat = np.clip(0.5 * BIAS_LAMBDA * B[0], -BIAS_CLIP, BIAS_CLIP)
    if BIAS_VAC_GUARD:
        vac_pri, _, _ = _vac_corr(state['hist'], state['vac_v'][:state['L']], state['vac_typ'][:state['L']], state['vac_v'][state['L']:state['L'] + H], state['vac_typ'][state['L']:state['L'] + H], state['past_dow'], state['past_hol'], state['profiles'], vlam)
        bias_hat = np.where(np.abs(vac_pri).max(axis=1) > 0.1, 0.0, bias_hat)
    gs_tbl = []
    combos = [(g, lo, hi) for g in GAMMAS for lo in SPREADS_LO for hi in SPREADS]
    default_ci = combos.index((0.0, 1.0, 1.0))
    for meta in aux_meta:
        m = meta['m']
        tab = np.zeros((N, len(combos)))
        for i in range(N):
            f0 = _corr_factors(meta['profs'][i], meta['out']['point'][i], meta['fdow'], meta['fhol'], beta[i])
            f0 = f0 * np.exp(meta['vac_by_lam'][VLAMS.index(vlam[i])][i])
            for ci, (g, lo, hiS) in enumerate(combos):
                fac = f0[:m] * np.exp(g * meta['rhat'][i, :m])
                qb = np.sort(meta['out']['quantiles'][i, :m] * fac[:, None], -1)
                tab[i, ci] = _pinball(meta['truth'][i], _spread(qb, lo, hiS))
        gs_tbl.append(tab)
    if gs_tbl:
        warr = np.array(wts)
        gl = np.array(gs_tbl)
        qual = warr >= QUALITY_FLOOR
        if qual.sum() < 2:
            qual = np.ones(len(warr), bool)
        glq, wq = (gl[qual], warr[qual])
        mean = (glq * wq[:, None, None]).sum(0) / wq.sum()
        base = np.maximum(mean[:, default_ci], 1e-09)
        pen = np.array([0.01 * abs(g) + 0.008 * (abs(lo - 1) + abs(hi - 1)) for g, lo, hi in combos])
        obj = mean / base[:, None] + pen[None, :]
        passing = np.all(glq <= glq[:, :, [default_ci]] * SPREAD_PASS, axis=0)
        passing[:, default_ci] = True
        obj = np.where(passing, obj, np.inf)
        pick = np.argmin(obj, axis=1)
        gamma = np.array([combos[c][0] for c in pick])
        sp_lo = np.array([combos[c][1] for c in pick])
        sp_hi = np.array([combos[c][2] for c in pick])
        if state.get('weather') is None:
            gamma[:] = 0.0
    else:
        gamma = np.zeros(N)
        sp_lo = np.ones(N)
        sp_hi = np.ones(N)
    spec_w = np.zeros(N)
    spec_tbl = []
    if SPEC_ENABLE and aux_meta:
        for meta in aux_meta:
            aux = meta['aux']
            m = meta['m']
            out = meta['out']
            sp_aux, _ = _spec_points(state['hist'][:, :aux['end']], aux['dows'], aux['hols'], aux['v_past'], aux['typ_past'], aux['v_fut'], aux['typ_fut'], aux['fdow'], aux['fhol'], aux['Xp'], aux['Xf'], H)
            tab = np.full((N, len(SPEC_WS)), np.nan)
            for i in range(N):
                f0 = _corr_factors(meta['profs'][i], out['point'][i], meta['fdow'], meta['fhol'], beta[i])
                f0 = f0 * np.exp(meta['vac_by_lam'][VLAMS.index(vlam[i])][i])
                f0 = f0 * np.exp(gamma[i] * meta['rhat'][i])
                pt = out['point'][i] * f0
                qb = _spread(np.sort(out['quantiles'][i] * f0[:, None], -1), sp_lo[i], sp_hi[i])
                for wi, wv in enumerate(SPEC_WS):
                    if wv > 0 and np.isfinite(sp_aux[i]).all():
                        ratio = np.exp((1 - wv) * np.log(np.maximum(pt, 1e-06)) + wv * sp_aux[i]) / np.maximum(pt, 1e-06)
                    else:
                        ratio = np.ones(H)
                    qq = np.sort(qb[:m] * ratio[:m, None], -1)
                    tab[i, wi] = _pinball(aux['truth'][i], qq)
            spec_tbl.append(tab)
        T = np.array(spec_tbl)
        Wv = np.array(wts)
        meanl = (T * Wv[:, None, None]).sum(0) / Wv.sum()
        rel = meanl / np.maximum(meanl[:, 0:1], 1e-09)
        pen = np.array([0.012 * wv / 0.5 for wv in SPEC_WS])
        worst = (T / np.maximum(T[:, :, 0:1], 1e-09)).max(0)
        obj = np.where(worst <= 1.15, rel + pen[None, :], np.inf)
        obj[:, 0] = 1.0
        spec_w = np.array(SPEC_WS)[np.argmin(obj, axis=1)]
    L = state['L']
    vac_f, vds, vdw = _vac_corr(state['hist'], state['vac_v'][:L], state['vac_typ'][:L], state['vac_v'][L:L + H], state['vac_typ'][L:L + H], state['past_dow'], state['past_hol'], state['profiles'], vlam)
    point = np.empty((N, H))
    quantiles = np.empty((N, H, 9))
    fac_used = np.ones((N, H))
    rhat_f = state['weather'](state['Xf'], state['anchor']) if state.get('weather') is not None else np.zeros((N, H))
    exit_floor = np.full((N, H), np.nan)
    if VAC_EXIT_FLOOR:
        vscal = state['vac_v'][:L]
        k28 = min(L, 28)
        wA = 0.5 ** ((k28 - 1 - np.arange(k28)) / VAC_ANCHOR_HL)
        anchor_v = np.sum(wA * vscal[L - k28:L]) / max(np.sum(wA), 1e-09)
        exit_day = anchor_v - state['vac_v'][L:L + H] > VAC_EXIT_THR
        if exit_day.any():
            lvl = _vac_exit_floor(state['hist'], vscal, state['vac_typ'][:L], state['vac_v'][L:L + H], state['past_dow'], state['past_hol'], state['profiles'], anchor_v)
            fd6 = np.where(state['fut_hol'], 6, state['fut_dow'])
            for i in range(N):
                if np.isfinite(lvl[i]) and vlam[i] > 0:
                    fl = np.exp(lvl[i] + state['profiles'][i][fd6])
                    day_i = exit_day & (vac_f[i] > 0.1)
                    exit_floor[i] = np.where(day_i, VAC_FLOOR_SHRINK * fl, np.nan)
    spec_log, spec_info = (None, [])
    if SPEC_ENABLE:
        spec_log, spec_info = _spec_points(state['hist'], state['past_dow'], state['past_hol'], state['vac_v'][:L], state['vac_typ'][:L], state['vac_v'][L:L + H], state['vac_typ'][L:L + H], state['fut_dow'], state['fut_hol'], state.get('Xp'), state.get('Xf'), H)
    for i in range(N):
        f = _corr_factors(state['profiles'][i], primary['point'][i], state['fut_dow'], state['fut_hol'], beta[i])
        if BIAS_LEAD_HL > 0:
            bias_lead = bias_hat[i] * 0.5 ** (np.arange(H) / BIAS_LEAD_HL)
        else:
            bias_lead = np.full(H, bias_hat[i])
        f = f * np.exp(vac_f[i]) * np.exp(-bias_lead) * np.exp(gamma[i] * rhat_f[i])
        fac_used[i] = f
        point[i] = primary['point'][i] * f
        qi = _spread(np.sort(primary['quantiles'][i] * f[:, None], axis=-1), sp_lo[i], sp_hi[i])
        quantiles[i] = _vac_widen(qi, vac_f[i])
        if VAC_EXIT_FLOOR and np.isfinite(exit_floor[i]).any():
            lift = np.maximum(exit_floor[i], point[i]) / np.maximum(point[i], 1e-09)
            lift = np.where(np.isfinite(exit_floor[i]), lift, 1.0)
            point[i] = point[i] * lift
            quantiles[i] = np.sort(quantiles[i] * lift[:, None], axis=-1)
        if spec_log is not None and spec_w[i] > 0 and np.isfinite(spec_log[i]).all():
            lp = (1 - spec_w[i]) * np.log(np.maximum(point[i], 1e-06)) + spec_w[i] * spec_log[i]
            ratio = np.exp(lp) / np.maximum(point[i], 1e-06)
            point[i] = point[i] * ratio
            quantiles[i] = np.sort(quantiles[i] * ratio[:, None], axis=-1)
    return {'point': point, 'quantiles': quantiles, 'components': {'mechanism': 'recency dow residual + holiday->Sunday + sign-consistent aux level-bias + vacation-calendar regime correction (probe 1)', 'beta_by_target': beta.tolist(), 'vac_lambda_by_target': vlam.tolist(), 'vac_delta_summer_by_target': vds.tolist(), 'vac_delta_winter_by_target': vdw.tolist(), 'vac_corr_range': [float(vac_f.min()), float(vac_f.max())], 'vac_active_horizon_days': int((np.abs(vac_f).max(axis=0) > 0.02).sum()), 'vac_informative_aux_by_target': np.array(vlam_info).sum(axis=0).tolist() if vlam_tbl else [], 'vac_native_pinball': [t.tolist() for t in vlam_tbl], 'spec_weight_by_target': spec_w.tolist(), 'spec_native_pinball': [t.tolist() for t in spec_tbl], 'spec_fit_info': spec_info, 'spec_trained': bool(SPEC_ENABLE and spec_log is not None), 'vac_exit_floor_days': int(np.isfinite(exit_floor).any(axis=0).sum()), 'vac_exit_floor_lift_max': float(np.nanmax(np.where(np.isfinite(exit_floor), exit_floor / np.maximum(point, 1e-09), np.nan))) if np.isfinite(exit_floor).any() else 1.0, 'gamma_by_target': gamma.tolist(), 'spread_lo_by_target': sp_lo.tolist(), 'spread_hi_by_target': sp_hi.tolist(), 'weather_rhat_range': [float(rhat_f.min()), float(rhat_f.max())], 'weather_model_trained': state.get('weather') is not None, 'bias_hat_by_target': bias_hat.tolist(), 'aux_origins': [a['offset'] for a in state['aux']], 'aux_weights_context_similarity': [a['weight'] for a in state['aux']], 'native_pinball_beta': [t.tolist() for t in beta_tbl], 'native_logbias_after_dow': [t.tolist() for t in bias_tbl], 'holiday_days_in_horizon': int(state['fut_hol'].sum()), 'factor_range': [float(fac_used.min()), float(fac_used.max())], 'actual_decision_from_native_errors': bool(beta_tbl), 'imputed_input_days_by_target': [int((~np.isfinite(state['hist'][i])).sum()) for i in range(N)]}}

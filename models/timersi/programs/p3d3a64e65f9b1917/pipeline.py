import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
LEVELS = np.arange(1, 10) / 10.0
BETAS = (0.0, 0.7, 1.0)
GAMMAS = (0.0, 0.5, 1.0)
SPREADS = (0.75, 0.85, 1.0)
SPREADS_LO = (0.7, 0.85, 1.0)
QUALITY_FLOOR = 0.0
RIDGE_ALPHA = 5.0
RHAT_CLIP = 0.3
WEATHER_COLS = ('AH', 'RH', 'T')
DEFAULT_BETA = 0.7
HALF_LIFE = 28.0
BIAS_LAMBDA = 0.5
BIAS_CLIP = 0.12
HOL_SHRINK = 0.8
MIN_FIT_DAYS = 21
HOLIDAYS = {'2004-04-12', '2004-04-25', '2004-05-01', '2004-06-02', '2004-08-15', '2004-11-01', '2004-12-08', '2004-12-25', '2004-12-26', '2005-01-01', '2005-01-06', '2005-03-28', '2005-04-25', '2005-05-01'}

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
        state['aux'].append({'offset': off, 'matured': m, 'end': end, 'truth': hist[:, end:end + m], 'dows': state['past_dow'][:end], 'hols': state['past_hol'][:end], 'fdow': _dow(ts[end:end + H]), 'fhol': _is_hol(ts[end:end + H]), 'weight': end / float(L), 'Xp': np.asarray(kf, float)[state['widx']][:, :end].T if state['widx'] else None, 'Xf': np.asarray(kf, float)[state['widx']][:, end:end + H].T if state['widx'] else None, 'anchor': np.nanmean(np.asarray(kf, float)[state['widx']][:, max(0, end - 14):end], axis=1) if state['widx'] else None})
    return (cases, state)

def postprocess(outputs, cases, state):
    N, H = (state['N'], state['H'])
    primary = outputs[0]
    beta_tbl, bias_tbl, wts, aux_meta = ([], [], [], [])
    for j, aux in enumerate(state['aux']):
        out = outputs[1 + j]
        m = aux['matured']
        if aux['end'] < MIN_FIT_DAYS:
            continue
        profs = _profiles(state['hist'][:, :aux['end']], aux['dows'], aux['hols'])
        pb = np.zeros((N, len(BETAS)))
        bias = np.zeros(N)
        for i in range(N):
            for bi, b in enumerate(BETAS):
                f = _corr_factors(profs[i], out['point'][i], aux['fdow'], aux['fhol'], b)
                qc = np.sort(out['quantiles'][i, :m] * f[:m, None], axis=-1)
                pb[i, bi] = _pinball(aux['truth'][i], qc)
            fb = _corr_factors(profs[i], out['point'][i], aux['fdow'], aux['fhol'], DEFAULT_BETA)
            pc = out['point'][i, :m] * fb[:m]
            bias[i] = np.nanmean(np.log(np.maximum(pc, 1e-06)) - np.log(np.maximum(aux['truth'][i], 1e-06)))
        wfit = _weather_fit(state['hist'][:, :aux['end']], aux['dows'], aux['hols'], profs, aux['Xp']) if aux['Xp'] is not None else None
        rhat = wfit(aux['Xf'], aux['anchor']) if wfit is not None else np.zeros((N, H))
        aux_meta.append({'out': out, 'm': m, 'truth': aux['truth'], 'profs': profs, 'rhat': rhat, 'fdow': aux['fdow'], 'fhol': aux['fhol']})
        beta_tbl.append(pb)
        bias_tbl.append(bias)
        wts.append(aux['weight'])
    if beta_tbl:
        w = np.array(wts)[:, None, None]
        loss = np.array(beta_tbl)
        scale = np.maximum((loss[..., 0] * np.array(wts)[:, None]).sum(0) / sum(wts), 1e-09)
        obj = (loss / scale[None, :, None] * w).sum(0) / sum(wts)
        obj = obj + 0.01 * np.abs(np.array(BETAS) - DEFAULT_BETA)[None, :]
        beta = np.array(BETAS)[np.argmin(obj, axis=-1)]
    else:
        beta = np.full(N, DEFAULT_BETA)
    bias_hat = np.zeros(N)
    if bias_tbl:
        B = np.array(bias_tbl)
        W = np.array(wts)
        qual = W >= QUALITY_FLOOR
        if qual.sum() >= 2:
            Bq, Wq = (B[qual], W[qual])
        else:
            Bq, Wq = (B, W)
        if Bq.shape[0] >= 2:
            same_sign = np.all(Bq > 0, axis=0) | np.all(Bq < 0, axis=0)
            lam = BIAS_LAMBDA if Bq.shape[0] == 2 else 0.7
            wmean = (Bq * Wq[:, None]).sum(0) / Wq.sum()
            bias_hat = np.where(same_sign, np.clip(lam * wmean, -BIAS_CLIP, BIAS_CLIP), 0.0)
        else:
            bias_hat = np.clip(0.5 * BIAS_LAMBDA * Bq[0], -BIAS_CLIP, BIAS_CLIP)
    gs_tbl = []
    combos = [(g, lo, hi) for g in GAMMAS for lo in SPREADS_LO for hi in SPREADS]
    default_ci = combos.index((0.0, 1.0, 1.0))
    for meta in aux_meta:
        m = meta['m']
        tab = np.zeros((N, len(combos)))
        for i in range(N):
            f0 = _corr_factors(meta['profs'][i], meta['out']['point'][i], meta['fdow'], meta['fhol'], beta[i])
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
        passing = np.all(glq <= glq[:, :, [default_ci]] * 1.002, axis=0)
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
    point = np.empty((N, H))
    quantiles = np.empty((N, H, 9))
    fac_used = np.ones((N, H))
    rhat_f = state['weather'](state['Xf'], state['anchor']) if state.get('weather') is not None else np.zeros((N, H))
    for i in range(N):
        f = _corr_factors(state['profiles'][i], primary['point'][i], state['fut_dow'], state['fut_hol'], beta[i])
        f = f * np.exp(-bias_hat[i]) * np.exp(gamma[i] * rhat_f[i])
        fac_used[i] = f
        point[i] = primary['point'][i] * f
        quantiles[i] = _spread(np.sort(primary['quantiles'][i] * f[:, None], axis=-1), sp_lo[i], sp_hi[i])
    return {'point': point, 'quantiles': quantiles, 'components': {'mechanism': 'recency dow residual + holiday->Sunday + sign-consistent aux level-bias', 'beta_by_target': beta.tolist(), 'gamma_by_target': gamma.tolist(), 'spread_lo_by_target': sp_lo.tolist(), 'spread_hi_by_target': sp_hi.tolist(), 'weather_rhat_range': [float(rhat_f.min()), float(rhat_f.max())], 'weather_model_trained': state.get('weather') is not None, 'bias_hat_by_target': bias_hat.tolist(), 'aux_origins': [a['offset'] for a in state['aux']], 'aux_weights_context_similarity': [a['weight'] for a in state['aux']], 'native_pinball_beta': [t.tolist() for t in beta_tbl], 'native_logbias_after_dow': [t.tolist() for t in bias_tbl], 'holiday_days_in_horizon': int(state['fut_hol'].sum()), 'factor_range': [float(fac_used.min()), float(fac_used.max())], 'actual_decision_from_native_errors': bool(beta_tbl), 'imputed_input_days_by_target': [int((~np.isfinite(state['hist'][i])).sum()) for i in range(N)]}}

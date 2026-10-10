import os
import sys
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
try:
    from loadspec import specialist_forecast
except Exception:
    specialist_forecast = None
NATIVE_WEIGHT = 0.6
SPREAD_POWER = 0.75
TEMP_KEYS = ('airtemperature', 'temperature', 'temp')
AUX_OFFSETS = (168, 336, 504, 672, 840, 1008)
SPREAD_GRID = np.arange(0.7, 1.31, 0.025)
SPREAD_DAMP = 0.6

def pd_roll_mean(v, w):
    out = np.empty_like(v, dtype=float)
    c = np.cumsum(np.insert(np.asarray(v, float), 0, 0.0))
    for i in range(len(v)):
        a = max(0, i - w + 1)
        out[i] = (c[i + 1] - c[a]) / (i + 1 - a)
    return out

def _find_temp(known_names):
    for i, nm in enumerate(known_names):
        if str(nm).strip().lower() in TEMP_KEYS:
            return i
    return None

def preprocess(view, card):
    state = {'N': len(view['target_ids']), 'H': view['horizon'], 'diag': {}}
    return (view, state)

def engineer(prepared, card, state):
    H = int(prepared['horizon'])
    L = int(prepared['cutoff_index'])
    names = list(prepared.get('known_names') or [])
    kf_raw = prepared.get('known_features')
    kf = np.asarray(kf_raw, dtype=float) if kf_raw is not None and len(kf_raw) else np.zeros((0, 0))
    ts = prepared['timestamps']
    hist = np.asarray(prepared['target_history'], dtype=float)
    obs = np.asarray(prepared.get('target_observed', np.ones_like(hist)), dtype=float)
    hist = np.where(obs > 0, hist, np.nan)
    ti = _find_temp(names)
    spec = np.full((state['N'], H), np.nan)
    diag = {}
    if specialist_forecast is not None and ti is not None and kf.size:
        temp_all = kf[ti]
        if temp_all.shape[0] >= L + H:
            temp_all = temp_all[:L + H]
            for d in range(state['N']):
                try:
                    p, info = specialist_forecast(ts[:L + H], hist[d][:L], temp_all, H)
                except Exception as exc:
                    p, info = (None, {'reason': 'exception:%s' % type(exc).__name__})
                if p is not None:
                    spec[d] = p
                diag[str(d)] = info
    else:
        diag['global'] = {'reason': 'no_temperature_or_module'}
    spec_hist = {}
    if specialist_forecast is not None and ti is not None and kf.size and (temp_all.shape[0] >= L):
        for off in AUX_OFFSETS:
            end = L - int(off)
            if end < 24 * 150 or end + H > L:
                continue
            ph = np.full((state['N'], H), np.nan)
            ok = True
            for d in range(state['N']):
                try:
                    p, _ = specialist_forecast(ts[:end + H], hist[d][:end], temp_all[:end + H], H)
                except Exception:
                    p = None
                if p is None:
                    ok = False
                    break
                ph[d] = p
            if ok:
                spec_hist[int(off)] = ph
    state['spec_hist'] = spec_hist
    try:
        from loadspec import balance_point
        state['balance_point'] = float(balance_point(ts[:L], hist[0][:L], kf[ti][:L]))
    except Exception:
        state['balance_point'] = 65.0
    state['spec'] = spec
    state['diag'] = diag
    return (prepared, state)

def _enriched_known(variables):
    pf = np.asarray(variables['known_features'], dtype=float) if len(variables['known_features']) else np.zeros((0, 0))
    names = list(variables['known_names'] or [])
    extra, extra_names = ([], [])
    ti = _find_temp(names)
    if ti is not None and pf.size:
        t = pf[ti]
        extra += [np.maximum(t - 65.0, 0.0), np.maximum(65.0 - t, 0.0), pd_roll_mean(t, 24)]
        extra_names += ['cooling_degree_h', 'heating_degree_h', 'temp_mean_24h']
    try:
        from loadspec import calendar_frame
        cf = calendar_frame(variables['timestamps'])
        extra += [cf['hol'].to_numpy(float), (cf['dow'].to_numpy() >= 5).astype(float)]
        extra_names += ['rule_holiday', 'is_weekend']
    except Exception:
        pass
    if extra:
        pf = np.vstack([pf, np.asarray(extra, dtype=float)]) if pf.size else np.asarray(extra, float)
        names = names + extra_names
    return (pf, names)

def _balance_known(variables, bp):
    pf = np.asarray(variables['known_features'], dtype=float) if len(variables['known_features']) else np.zeros((0, 0))
    names = list(variables['known_names'] or [])
    ti = _find_temp(names)
    if ti is None or not pf.size:
        return (pf, names)
    t = pf[ti]
    extra = [np.maximum(t - bp, 0.0), np.maximum(bp - t, 0.0), pd_roll_mean(t, 24)]
    enames = ['cooling_degree_h', 'heating_degree_h', 'temp_mean_24h']
    try:
        from loadspec import calendar_frame
        cf = calendar_frame(variables['timestamps'])
        extra += [cf['hol'].to_numpy(float), (cf['dow'].to_numpy() >= 5).astype(float)]
        enames += ['rule_holiday', 'is_weekend']
    except Exception:
        pass
    return (np.vstack([pf, np.asarray(extra, float)]), names + enames)

def _minimal_known(variables):
    pf = np.asarray(variables['known_features'], dtype=float) if len(variables['known_features']) else np.zeros((0, 0))
    names = list(variables['known_names'] or [])
    ti = _find_temp(names)
    if ti is None or not pf.size:
        return (pf, names)
    t = pf[ti]
    return (np.vstack([pf, np.maximum(t - 65.0, 0.0), np.maximum(65.0 - t, 0.0)]), names + ['cooling_degree_h', 'heating_degree_h'])

def select_context(variables, card, state):
    L = int(variables['cutoff_index'])
    H = int(variables['horizon'])
    maxC = int(card['limits']['max_context'])
    C = min(L, maxC)
    po = variables['past_features']
    raw_pf = np.asarray(variables['known_features'], dtype=float) if len(variables['known_features']) else np.zeros((0, 0))
    pf, names = _enriched_known(variables)
    bp = state.get('balance_point', 65.0)
    pf_bal, names_bal = _balance_known(variables, bp)
    pf_min, names_min = _minimal_known(variables)
    raw_names = list(variables['known_names'] or [])
    hist = np.asarray(variables['target_history'], dtype=float)
    cases = [{'target_indices': list(range(state['N'])), 'targets': hist[:, -C:], 'past_only': po[:, -C:] if len(po) else None, 'known_future': pf[:, -(C + H):] if pf.size else None, 'past_names': variables['past_names'], 'known_names': names, 'provenance': 'Primary: reference target/context + degree-hour and calendar known covariates'}]
    aux_meta = []
    for off in AUX_OFFSETS:
        end = L - int(off)
        if end < 24 * 150 or end + H > L:
            continue
        Ca = min(end, maxC)
        variants = [('enriched', pf, names), ('balance%d' % int(bp), pf_bal, names_bal), ('minimal', pf_min, names_min)]
        for tag, feats, fnames in variants:
            kf = feats[:, end - Ca:end + H] if getattr(feats, 'size', 0) else None
            cases.append({'target_indices': list(range(state['N'])), 'targets': hist[:, end - Ca:end], 'past_only': po[:, end - Ca:end] if len(po) else None, 'known_future': kf, 'past_names': variables['past_names'], 'known_names': fnames if kf is not None else raw_names, 'auxiliary': True, 'origin_offset': int(off), 'provenance': 'Auxiliary native backtest at origin L-%d (%s known set)' % (off, tag)})
            aux_meta.append({'offset': int(off), 'variant': tag, 'context': int(Ca), 'end': int(end), 'truth': hist[:, end:end + H].tolist()})
    state['aux_meta'] = aux_meta
    state['scale'] = _mase_scale(hist, int(card.get('seasonality') or 24))
    return (cases, state)
_QL = np.arange(1, 10) / 10.0

def _sql(truth, q, s):
    e = np.asarray(truth, float)[:, None] - np.asarray(q, float)
    return float(2.0 * np.mean(np.maximum(_QL * e, (_QL - 1.0) * e)) / max(s, 1e-09))

def _mase_scale(hist, season=24):
    out = []
    for row in np.asarray(hist, dtype=float):
        d = np.abs(row[season:] - row[:-season])
        d = d[np.isfinite(d)]
        out.append(float(np.mean(d)) if d.size else 1.0)
    return np.asarray(out, float)

def postprocess(outputs, cases, state):
    N, H = (state['N'], state['H'])
    point = np.empty((N, H))
    quantiles = np.empty((N, H, 9))
    aux = []
    for output, case in zip(outputs, cases):
        if case.get('auxiliary'):
            aux.append((case, output))
            continue
        point[case['target_indices']] = output['point']
        quantiles[case['target_indices']] = output['quantiles']
    scl = state.get('scale', np.ones(N))
    meta = state.get('aux_meta', [])
    spec_hist = state.get('spec_hist', {})
    report = []
    cal = {d: [] for d in range(N)}
    for (case, output), m in zip(aux, meta):
        tr = np.asarray(m['truth'], float)
        pt = np.asarray(output['point'], float)
        qq = np.asarray(output['quantiles'], float)
        if tr.shape != pt.shape:
            continue
        ent = {'offset': m['offset'], 'variant': m['variant']}
        ent['native_mase'] = [float(np.mean(np.abs(tr[d] - pt[d])) / max(scl[d], 1e-09)) for d in range(tr.shape[0])]
        ent['native_sql'] = [_sql(tr[d], qq[d], scl[d]) for d in range(tr.shape[0])]
        ent['native_rel_bias'] = [float(np.mean(tr[d] - pt[d]) / max(abs(np.mean(tr[d])), 1e-09)) for d in range(tr.shape[0])]
        sp = spec_hist.get(m['offset'])
        if sp is not None:
            sp = np.asarray(sp, float)
            bl = NATIVE_WEIGHT * pt + (1.0 - NATIVE_WEIGHT) * sp
            ent['spec_mase'] = [float(np.mean(np.abs(tr[d] - sp[d])) / max(scl[d], 1e-09)) for d in range(tr.shape[0])]
            ent['blend_mase'] = [float(np.mean(np.abs(tr[d] - bl[d])) / max(scl[d], 1e-09)) for d in range(tr.shape[0])]
            ent['err_corr'] = [float(np.corrcoef(tr[d] - pt[d], tr[d] - sp[d])[0, 1]) for d in range(tr.shape[0])]
            if m['variant'] == 'enriched':
                for d in range(tr.shape[0]):
                    cal[d].append((tr[d], pt[d], qq[d], sp[d]))
        report.append(ent)
    state['diag']['native_backtest'] = report
    spread_factor = {}
    for d in range(N):
        if len(cal[d]) < 2:
            continue
        best, best_loss = (1.0, None)
        for fct in SPREAD_GRID:
            loss = 0.0
            for tr, pt, qq, sp in cal[d]:
                bl = NATIVE_WEIGHT * pt + (1.0 - NATIVE_WEIGHT) * sp
                q = np.sort(bl[:, None] + (qq - pt[:, None]) * fct, axis=-1)
                loss += _sql(tr, q, scl[d])
            if best_loss is None or loss < best_loss:
                best, best_loss = (float(fct), loss)
        spread_factor[d] = 1.0 + SPREAD_DAMP * (best - 1.0)
        state['diag'].setdefault('measured_spread', {})[str(d)] = {'argmin': best, 'applied': spread_factor[d], 'origins': len(cal[d])}
    state['spread_factor'] = spread_factor
    spec = state.get('spec')
    used = 0
    if spec is not None:
        for d in range(N):
            s = spec[d]
            if s is None or not np.all(np.isfinite(s)):
                continue
            nat = point[d].copy()
            spread = quantiles[d] - nat[:, None]
            new = NATIVE_WEIGHT * nat + (1.0 - NATIVE_WEIGHT) * s
            f = float(state.get('spread_factor', {}).get(d, 1.0))
            point[d] = new
            quantiles[d] = new[:, None] + spread * f
            state['diag'].setdefault('spread_factor', {})[str(d)] = f
            used += 1
    state['diag']['targets_combined'] = used
    quantiles = np.sort(quantiles, axis=-1)
    return {'point': point, 'quantiles': quantiles, 'components': {'native_weight': NATIVE_WEIGHT, 'diagnostics': state['diag']}}

import os
import sys
import numpy as np
import pandas as pd
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
try:
    from loadspec import specialist_forecast, any_specialist, any_specialist_variants
except Exception:
    specialist_forecast = None
    any_specialist = None
    any_specialist_variants = None
NATIVE_WEIGHT = 0.6
SPREAD_POWER = 0.75
TEMP_KEYS = ('airtemperature', 'temperature', 'temp')
AUX_OFFSETS = (168, 336, 504, 672, 840, 1008)
AUX_EXTRA = (1176, 1344)
AUX_YEAR = (8400, 8568, 8736, 8904)
SPREAD_GRID = np.arange(0.7, 1.31, 0.025)
SPREAD_DAMP = 0.8
GUARD_SHRINK = 1.6
GUARD_DROP = 2.5
MIN_SPEC_CONTEXT = 336
FULL_SPEC_CONTEXT = 24 * 120

def weight_from_ratio(ratio, mode, measured):
    if mode == 'short':
        if not measured:
            return 0.15
        if ratio <= 1.0:
            return 0.3
        if ratio <= 1.5:
            return 0.15
        return 0.0
    if not measured:
        return 1.0 - NATIVE_WEIGHT
    if ratio <= 0.9:
        return 0.5
    if ratio <= 1.15:
        return 0.4
    if ratio <= 1.5:
        return 0.3
    if ratio <= 2.0:
        return 0.15
    return 0.0

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
    spec = {}
    diag = {}
    state['spec_mode'] = 'none'
    temp_all = None
    if any_specialist_variants is not None and ti is not None and kf.size:
        temp_all = kf[ti]
        if temp_all.shape[0] >= L + H:
            temp_all = temp_all[:L + H]
            for d in range(state['N']):
                try:
                    p, info, mode = any_specialist_variants(ts[:L + H], hist[d][:L], temp_all, H)
                except Exception as exc:
                    p, info, mode = (None, {'reason': 'exception:%s' % type(exc).__name__}, 'none')
                if p:
                    spec[d] = p
                    state['spec_mode'] = mode
                diag[str(d)] = dict(info, mode=mode)
    else:
        diag['global'] = {'reason': 'no_temperature_or_module'}
    spec_hist = {}
    if any_specialist is not None and ti is not None and kf.size and (temp_all is not None) and (temp_all.shape[0] >= L):
        offsets = [int(o) for o in AUX_OFFSETS if L - int(o) >= MIN_SPEC_CONTEXT]
        offsets += [int(o) for o in AUX_EXTRA if L - int(o) >= FULL_SPEC_CONTEXT]
        offsets += [int(o) for o in AUX_YEAR if L - int(o) >= FULL_SPEC_CONTEXT]
        if not offsets and L > MIN_SPEC_CONTEXT:
            offsets = [min(int(L - MIN_SPEC_CONTEXT), H - 1)]
        for off in offsets:
            end = L - int(off)
            ph = {}
            ok = True
            for d in range(state['N']):
                try:
                    p, _, _ = any_specialist_variants(ts[:end + H], hist[d][:end], temp_all[:end + H], H)
                except Exception:
                    p = None
                if not p:
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
    try:
        from loadspec import _extreme_day_mask
        if ti is not None and kf.size and (temp_all is not None) and (temp_all.shape[0] >= L + H) and (state.get('spec_mode') == 'full'):
            state['extreme_mask'] = _extreme_day_mask(temp_all, L, H)
        else:
            state['extreme_mask'] = np.zeros(H, bool)
    except Exception:
        state['extreme_mask'] = np.zeros(H, bool)
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
        import loadspec as _ls
        from loadspec import calendar_frame
        _sav = _ls._HOL_VALID
        _ls._HOL_VALID = None
        try:
            cf = calendar_frame(variables['timestamps'])
        finally:
            _ls._HOL_VALID = _sav
        extra += [cf['hol'].to_numpy(float), (cf['dow'].to_numpy() >= 5).astype(float)]
        extra_names += ['rule_holiday', 'is_weekend']
    except Exception:
        pass
    if extra:
        pf = np.vstack([pf, np.asarray(extra, dtype=float)]) if pf.size else np.asarray(extra, float)
        names = names + extra_names
    return (pf, names)

def select_context(variables, card, state):
    L = int(variables['cutoff_index'])
    H = int(variables['horizon'])
    maxC = int(card['limits']['max_context'])
    C = min(L, maxC)
    po = variables['past_features']
    raw_pf = np.asarray(variables['known_features'], dtype=float) if len(variables['known_features']) else np.zeros((0, 0))
    pf, names = _enriched_known(variables)
    raw_names = list(variables['known_names'] or [])
    hist = np.asarray(variables['target_history'], dtype=float)
    cases = [{'target_indices': list(range(state['N'])), 'targets': hist[:, -C:], 'past_only': po[:, -C:] if len(po) else None, 'known_future': pf[:, -(C + H):] if pf.size else None, 'past_names': variables['past_names'], 'known_names': names, 'provenance': 'Primary: reference target/context + degree-hour and calendar known covariates'}]
    aux_meta = []
    full_offsets = [int(o) for o in AUX_OFFSETS if L - int(o) >= MIN_SPEC_CONTEXT and L - int(o) + H <= L]
    full_offsets += [int(o) for o in AUX_EXTRA if L - int(o) >= FULL_SPEC_CONTEXT and L - int(o) + H <= L]
    full_offsets += [int(o) for o in AUX_YEAR if L - int(o) >= FULL_SPEC_CONTEXT and L - int(o) + H <= L]
    partial_offsets = []
    if not full_offsets and L > MIN_SPEC_CONTEXT:
        partial_offsets = [min(int(L - MIN_SPEC_CONTEXT), H - 1)]
    for off in full_offsets + partial_offsets:
        end = L - int(off)
        Ca = min(end, maxC)
        partial = off in partial_offsets
        feats, fnames = (pf[:, :end + H], names)
        kf = feats[:, end - Ca:end + H] if getattr(feats, 'size', 0) else None
        case = {'target_indices': list(range(state['N'])), 'targets': hist[:, end - Ca:end], 'past_only': po[:, end - Ca:end] if len(po) else None, 'known_future': kf, 'past_names': variables['past_names'], 'known_names': fnames if kf is not None else raw_names, 'auxiliary': True, 'origin_offset': int(off), 'provenance': 'Auxiliary native backtest at origin L-%d (%s)' % (off, 'partial' if partial else 'enriched')}
        if partial:
            case['partial_auxiliary'] = True
        cases.append(case)
        aux_meta.append({'offset': int(off), 'variant': 'enriched', 'context': int(Ca), 'end': int(end), 'partial': bool(partial), 'matured': int(off) if partial else int(H), 'truth': hist[:, end:min(end + H, L)].tolist()})
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
        mat = int(m.get('matured', tr.shape[-1]))
        pt = np.asarray(output['point'], float)[:, :mat]
        qq = np.asarray(output['quantiles'], float)[:, :mat]
        tr = tr[:, :mat]
        if tr.shape != pt.shape:
            continue
        ent = {'offset': m['offset'], 'variant': m['variant'], 'partial': bool(m.get('partial', False)), 'matured': mat}
        ent['native_mase'] = [float(np.mean(np.abs(tr[d] - pt[d])) / max(scl[d], 1e-09)) for d in range(tr.shape[0])]
        ent['native_sql'] = [_sql(tr[d], qq[d], scl[d]) for d in range(tr.shape[0])]
        ent['native_rel_bias'] = [float(np.mean(tr[d] - pt[d]) / max(abs(np.mean(tr[d])), 1e-09)) for d in range(tr.shape[0])]
        sp = spec_hist.get(m['offset'])
        if sp is not None:
            ent['spec_mase'] = {}
            for d in range(tr.shape[0]):
                vd = {nm: np.asarray(v, float)[:mat] for nm, v in sp[d].items()}
                ent['spec_mase'][str(d)] = {nm: round(float(np.mean(np.abs(tr[d] - v)) / max(scl[d], 1e-09)), 4) for nm, v in vd.items()}
                cal[d].append((tr[d], pt[d], qq[d], vd))
        report.append(ent)
    state['diag']['native_backtest'] = report
    weight, variant_pick = ({}, {})
    mode = state.get('spec_mode', 'none')
    W_GRID = np.arange(0.3, 0.91, 0.05)
    for d in range(N):
        entries = cal[d]
        measured = len(entries) >= 1
        if not measured:
            weight[d] = 1.0 - weight_from_ratio(float('nan'), mode, False)
            variant_pick[d] = 'recent'
            state['diag'].setdefault('measured_weight', {})[str(d)] = {'origins': 0, 'mode': mode, 'native_weight': weight[d], 'variant': 'recent'}
            continue
        vnames = set()
        for tr, pt, qq, vd in entries:
            vnames.update(vd.keys())
        vmae = {}
        for nm in vnames:
            es = [np.mean(np.abs(tr - vd[nm])) for tr, pt, qq, vd in entries if nm in vd]
            if es:
                vmae[nm] = float(np.mean(es))
        pick = 'recent' if 'recent' in vmae else sorted(vmae)[0]
        for nm, v in vmae.items():
            if v < 0.99 * vmae.get(pick, np.inf):
                pick = nm
        variant_pick[d] = pick
        en = float(np.mean([np.mean(np.abs(tr - pt)) for tr, pt, qq, vd in entries]))
        ratio = vmae.get(pick, np.inf) / max(en, 1e-09)
        if mode != 'full' or len(entries) < 3:
            w_spec = weight_from_ratio(ratio, mode, True)
            weight[d] = 1.0 - w_spec
        elif ratio > 2.0:
            weight[d] = 1.0
        else:
            best, best_loss = (NATIVE_WEIGHT, None)
            for w in W_GRID:
                losses = []
                for tr, pt, qq, vd in entries:
                    if pick not in vd:
                        continue
                    bl = w * pt + (1.0 - w) * vd[pick]
                    q = np.sort(bl[:, None] + (qq - pt[:, None]), axis=-1)
                    losses.append(_sql(tr, q, scl[d]))
                if not losses:
                    continue
                loss = float(np.mean(losses))
                if best_loss is None or loss < best_loss:
                    best, best_loss = (float(w), loss)
            w_d = NATIVE_WEIGHT + 0.8 * (best - NATIVE_WEIGHT)
            weight[d] = float(min(max(w_d, 0.35), 1.0))
        state['diag'].setdefault('measured_weight', {})[str(d)] = {'spec_over_native': round(float(ratio), 4), 'origins': len(entries), 'mode': mode, 'native_weight': round(weight[d], 4), 'variant': pick, 'variant_mae': {nm: round(v, 2) for nm, v in vmae.items()}}
    state['weight'] = weight
    state['variant_pick'] = variant_pick
    spread_factor = {}
    SIDE_GRID = np.arange(0.5, 1.31, 0.025)
    for d in range(N):
        full_cal = [c for c in cal[d] if c[0].shape[-1] >= state['H']]
        entries = full_cal
        damp = SPREAD_DAMP
        if len(full_cal) < 2:
            entries = [c for c in cal[d] if c[0].shape[-1] >= 24]
            damp = 0.4
            if not entries:
                continue
        w_d = float(state.get('weight', {}).get(d, NATIVE_WEIGHT))
        pick = state.get('variant_pick', {}).get(d, 'recent')
        chosen = {}
        for side in ('lo', 'up'):
            best, best_loss = (1.0, None)
            for fct in SIDE_GRID:
                losses = []
                for tr, pt, qq, vd in entries:
                    sp = vd.get(pick)
                    bl = pt if sp is None or w_d >= 1.0 else w_d * pt + (1.0 - w_d) * sp
                    dq = qq - pt[:, None]
                    f = np.where(dq < 0, fct if side == 'lo' else 1.0, fct if side == 'up' else 1.0)
                    q = np.sort(bl[:, None] + dq * f, axis=-1)
                    losses.append(_sql(tr, q, scl[d]))
                if not losses:
                    continue
                loss = float(np.mean(losses))
                if best_loss is None or loss < best_loss:
                    best, best_loss = (float(fct), loss)
            chosen[side] = 1.0 + damp * (best - 1.0)
            chosen[side + '_argmin'] = best
        if not chosen:
            continue
        spread_factor[d] = (float(np.clip(chosen['lo'], 0.6, 1.25)), float(np.clip(chosen['up'], 0.6, 1.25)))
        state['diag'].setdefault('measured_spread', {})[str(d)] = {'argmin_lo': chosen['lo_argmin'], 'argmin_up': chosen['up_argmin'], 'applied': list(spread_factor[d]), 'origins': len(entries), 'partial_based': len(full_cal) < 2}
    state['spread_factor'] = spread_factor
    emp_choice = {}
    emp_quantiles = {}
    for d in range(N):
        full_cal = [c for c in cal[d] if c[0].shape[-1] >= state['H']]
        if len(full_cal) < 4:
            continue
        w_d = float(state.get('weight', {}).get(d, NATIVE_WEIGHT))
        pick = state.get('variant_pick', {}).get(d, 'recent')
        flo, fup = state.get('spread_factor', {}).get(d, (1.0, 1.0))
        Hh = state['H']
        nblk = max(1, Hh // 24)

        def _blend(e):
            tr, pt, qq, vd = e
            sp = vd.get(pick)
            return pt if sp is None or w_d >= 1.0 else w_d * pt + (1.0 - w_d) * sp
        errs = [c[0] - _blend(c) for c in full_cal]
        houridx = np.arange(Hh) % 24

        def _emp_q(others, bl, bucket):
            qe = np.empty((Hh, 9))
            if bucket == 'day':
                for b in range(nblk):
                    a0, a1 = (b * 24, Hh if b == nblk - 1 else (b + 1) * 24)
                    qs = np.quantile(others[:, a0:a1].ravel(), _QL)
                    qe[a0:a1] = bl[a0:a1, None] + qs[None, :]
            else:
                for hh in range(24):
                    m = houridx == hh
                    qs = np.quantile(others[:, m].ravel(), _QL)
                    qe[m] = bl[m, None] + qs[None, :]
            return np.sort(qe, -1)
        loo = {'native_shape': [], 'day': [], 'hour': []}
        for i, c in enumerate(full_cal):
            tr, pt, qq, vd = c
            bl = _blend(c)
            others = np.vstack([errs[j] for j in range(len(full_cal)) if j != i])
            for bucket in ('day', 'hour'):
                loo[bucket].append(_sql(tr, _emp_q(others, bl, bucket), scl[d]))
            dq = qq - pt[:, None]
            qn = np.sort(bl[:, None] + dq * np.where(dq < 0, flo, fup), axis=-1)
            loo['native_shape'].append(_sql(tr, qn, scl[d]))
        meds = {m: float(np.median(v)) for m, v in loo.items()}
        best_emp = min(('day', 'hour'), key=lambda m: meds[m])
        use_emp = meds[best_emp] < 0.97 * meds['native_shape']
        emp_choice[d] = best_emp if use_emp else None
        state['diag'].setdefault('empirical_arm', {})[str(d)] = {'loo_medians': {m: round(v, 4) for m, v in meds.items()}, 'chosen': best_emp if use_emp else 'native_shape', 'origins': len(full_cal)}
        if use_emp:
            allerr = np.vstack(errs)
            if best_emp == 'day':
                qs_blk = np.empty((nblk, 9))
                for b in range(nblk):
                    a0, a1 = (b * 24, Hh if b == nblk - 1 else (b + 1) * 24)
                    qs_blk[b] = np.quantile(allerr[:, a0:a1].ravel(), _QL)
            else:
                qs_blk = np.empty((24, 9))
                for hh in range(24):
                    qs_blk[hh] = np.quantile(allerr[:, houridx == hh].ravel(), _QL)
            emp_quantiles[d] = (best_emp, qs_blk)
    state['emp_choice'] = emp_choice
    state['emp_quantiles'] = emp_quantiles

    def _apply_emp(d, base):
        bucket, qs_blk = state['emp_quantiles'][d]
        qe = np.empty((H, 9))
        if bucket == 'day':
            nblk = qs_blk.shape[0]
            for b in range(nblk):
                a0 = b * 24
                a1 = H if b == nblk - 1 else (b + 1) * 24
                qe[a0:a1] = base[a0:a1, None] + qs_blk[b][None, :]
        else:
            houridx = np.arange(H) % 24
            for hh in range(24):
                m = houridx == hh
                qe[m] = base[m, None] + qs_blk[hh][None, :]
        return np.sort(qe, axis=-1)
    spec = state.get('spec') or {}
    used = 0
    if True:
        for d in range(N):
            vd = spec.get(d)
            pick = state.get('variant_pick', {}).get(d, 'recent')
            s = None
            if vd:
                s = vd.get(pick)
                if s is None:
                    s = list(vd.values())[0]
            w = float(state.get('weight', {}).get(d, NATIVE_WEIGHT))
            flo, fup = state.get('spread_factor', {}).get(d, (1.0, 1.0))
            nat = point[d].copy()
            spread = quantiles[d] - nat[:, None]
            sided = spread * np.where(spread < 0, flo, fup)
            if s is None or not np.all(np.isfinite(s)) or w >= 1.0:
                if state.get('emp_choice', {}).get(d) and d in state.get('emp_quantiles', {}):
                    quantiles[d] = _apply_emp(d, nat)
                else:
                    quantiles[d] = np.sort(nat[:, None] + sided, axis=-1)
                state['diag'].setdefault('spread_factor', {})[str(d)] = [flo, fup]
                continue
            W_EXT = 0.35
            ext = np.asarray(state.get('extreme_mask', np.zeros(H, bool)), bool)
            w_vec = np.full(H, w)
            W_FAR_TILT = 0.1
            if H > 96:
                w_vec[96:] = max(w - W_FAR_TILT, 0.3)
            if ext.any():
                w_vec = np.where(ext, np.minimum(w, W_EXT), w_vec)
                state['diag'].setdefault('extreme_weight', {})[str(d)] = {'hours': int(ext.sum()), 'w_normal': round(w, 3), 'w_extreme': float(min(w, W_EXT))}
            new = w_vec * nat + (1.0 - w_vec) * s
            point[d] = new
            if state.get('emp_choice', {}).get(d) and d in state.get('emp_quantiles', {}):
                quantiles[d] = _apply_emp(d, new)
            else:
                quantiles[d] = new[:, None] + sided
            state['diag'].setdefault('spread_factor', {})[str(d)] = [flo, fup]
            if w < 1.0:
                used += 1
    state['diag']['targets_combined'] = used
    quantiles = np.sort(quantiles, axis=-1)
    return {'point': point, 'quantiles': quantiles, 'components': {'native_weight': NATIVE_WEIGHT, 'diagnostics': state['diag']}}

import numpy as np
from imputation import reconstruct
import backtest as bt
import specialist as sp
HALFLIFE_DAYS = 56
RECON_ITERS = 6
ARM_CLIP = (0.3, 0.7)
TAIL_COV_GATE = 0.885
TAIL_GAMMA = 0.9
SPEC_W_MAX = 0.65
SPEC_LR_CAP = 0.5
SPEC_MIN_CONTEXT = 6 * 168
AUX_RECENCY = {168: 0.65, 336: 0.35}

def _recon(th, ob, ts, kf, upto):
    sub_ob = ob[:, :upto]
    if sub_ob.all():
        return th[:, :upto].copy()
    return reconstruct(th[:, :upto], sub_ob, ts[:upto], kf[:, :upto] if kf.size else None, halflife_h=24 * HALFLIFE_DAYS, iters=RECON_ITERS)

def preprocess(view, card):
    prepared = dict(view)
    th = np.asarray(view['target_history'], float)
    ob = np.asarray(view['target_observed'], bool)
    L = int(view['cutoff_index'])
    state = {'N': len(view['target_ids']), 'H': int(view['horizon']), 'names': list(view['target_names']), 'L': L, 'missing_fraction': float(1.0 - ob.mean())}
    state['clean_history'] = th
    state['raw_history'] = th
    state['observed'] = ob
    if ob.all():
        state['reconstructed'] = False
        return (prepared, state)
    kf = np.asarray(view['known_features'], float)
    try:
        filled = _recon(th, ob, np.asarray(view['timestamps']), kf, L)
        if np.isfinite(filled).all() and (filled > 0).all():
            prepared['target_history'] = filled
            state['clean_history'] = filled
            state['reconstructed'] = True
        else:
            state['reconstructed'] = False
    except Exception as exc:
        state['reconstructed'] = False
        state['reconstruction_error'] = repr(exc)
    return (prepared, state)

def engineer(prepared, card, state):
    state['aux'] = []
    try:
        L, H = (state['L'], state['H'])
        ob = state['observed']
        th = np.asarray(state['raw_history'], float)
        ts = np.asarray(prepared['timestamps'])
        kf = np.asarray(prepared['known_features'], float)
        for off, frac in bt.pick_origins(ob, L, H):
            cut = L - off
            try:
                hist = _recon(th, ob, ts, kf, cut)
                if not (np.isfinite(hist).all() and (hist > 0).all()):
                    hist = th[:, :cut]
            except Exception:
                hist = th[:, :cut]
            state['aux'].append({'offset': int(off), 'frac': float(frac), 'cut': int(cut), 'hist': hist, 'truth': th[:, cut:cut + H].copy(), 'mask': ob[:, cut:cut + H].copy()})
    except Exception as exc:
        state['aux'] = []
        state['aux_error'] = repr(exc)
    state['spec'] = None
    try:
        L, H = (state['L'], state['H'])
        if L >= SPEC_MIN_CONTEXT:
            th = np.asarray(state['raw_history'], float)
            ob = state['observed']
            kf = np.asarray(prepared['known_features'], float)
            X = sp.features(prepared['timestamps'], kf, prepared['known_names'])
            names = list(prepared['known_names'])
            t_col = 8 + 2 * names.index('T') if 'T' in names else None
            spec = {'point': {}, 'sigma': {}, 'eS': {}, 'n_train': {}, 'support': 1.0}
            if t_col is not None:
                spec['support'] = sp.support_fraction(X[:L], X[L:L + H], t_col)
            for i in range(state['N']):
                fp = sp.fit_predict(X, th[i], ob[i], L, L, L + H)
                if fp is None:
                    continue
                spec['point'][i], spec['sigma'][i], spec['n_train'][i] = fp
                es = {}
                for a in state['aux']:
                    cut = a['cut']
                    bk = sp.fit_predict(X, th[i], ob[i], cut, cut, cut + H)
                    if bk is None:
                        continue
                    pb = sp.pinball(a['truth'][i], a['mask'][i], bk[0], bk[1])
                    if np.isfinite(pb):
                        es[a['offset']] = pb
                spec['eS'][i] = es
            if spec['point']:
                state['spec'] = spec
    except Exception as exc:
        state['spec'] = None
        state['spec_error'] = repr(exc)
    return (prepared, state)
ITALIAN_HOLIDAYS = {(1, 1): 1.0, (1, 6): 1.0, (4, 25): 1.0, (5, 1): 1.0, (6, 2): 1.0, (8, 15): 1.0, (11, 1): 1.0, (12, 8): 1.0, (12, 25): 1.0, (12, 26): 1.0, (12, 24): 0.5, (12, 31): 0.5}
EASTER = {(2004, 4, 11): 1.0, (2004, 4, 12): 1.0, (2005, 3, 27): 1.0, (2005, 3, 28): 1.0}

def _vacation(month, day):
    if month == 8:
        return 1.0 if 7 <= day <= 22 else 0.6
    return 0.0

def _semantic_covariates(timestamps, kf, known_names):
    import pandas as pd
    ts = pd.to_datetime(pd.Series(np.asarray(timestamps)))
    hour = ts.dt.hour.values.astype(float)
    dow = ts.dt.dayofweek.values.astype(float)
    hol = np.array([max(ITALIAN_HOLIDAYS.get((m, d), 0.0), EASTER.get((y, m, d), 0.0)) for y, m, d in zip(ts.dt.year, ts.dt.month, ts.dt.day)], float)
    vac = np.array([_vacation(m, d) for m, d in zip(ts.dt.month, ts.dt.day)], float)
    rows = [np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24), np.sin(4 * np.pi * hour / 24), np.cos(4 * np.pi * hour / 24), (dow >= 5).astype(float), hol, vac]
    names = ['cal_hsin1', 'cal_hcos1', 'cal_hsin2', 'cal_hcos2', 'cal_weekend', 'cal_holiday', 'cal_vacation']
    if kf is not None and kf.size:
        for j, nm in enumerate(list(known_names)):
            x = pd.Series(np.asarray(kf[j], float))
            rows.append(x.rolling(24, min_periods=1).mean().values)
            names.append('syn24_%s' % nm)
    return (np.asarray(rows, float), names)

def select_context(variables, card, state):
    L = int(variables['cutoff_index'])
    H = int(variables['horizon'])
    maxc = int(card['limits']['max_context'])
    C = int(min(L, maxc))
    po = np.asarray(variables['past_features'], float)
    pf_plain = np.asarray(variables['known_features'], float)
    use_semantic = L >= 6 * 168
    plain_names = list(variables['known_names'])
    sem_names = plain_names
    pf_sem = pf_plain
    state['semantic_covariates'] = []
    if use_semantic:
        try:
            extra, extra_names = _semantic_covariates(variables['timestamps'], pf_plain, plain_names)
            pf_sem = np.vstack([pf_plain, extra]) if pf_plain.size else extra
            sem_names = plain_names + extra_names
            state['semantic_covariates'] = extra_names
        except Exception as exc:
            state['semantic_error'] = repr(exc)
            use_semantic = False
    state['dual_arm'] = bool(use_semantic)
    th = np.asarray(variables['target_history'], float)
    idx = list(range(state['N']))

    def mk(targets, pfX, namesX, lo, hi, **flags):
        return {'target_indices': idx, 'targets': targets, 'past_only': po[:, lo:hi] if po.size else None, 'known_future': pfX[:, lo:hi + H] if pfX.size else None, 'past_names': list(variables['past_names']), 'known_names': namesX, **flags}
    cases = [dict(mk(th[:, L - C:L], pf_sem, sem_names, L - C, L), provenance='Primary arm A: reconstructed history + constructed calendar/holiday/synoptic covariates, context %d h' % C)]
    for a in state.get('aux', []):
        cut = a['cut']
        Ca = int(min(cut, maxc))
        flags = {'alternative': True, 'partial_backtest': True} if a.get('partial') else {'auxiliary': True}
        cases.append(dict(mk(np.asarray(a['hist'], float)[:, cut - Ca:cut], pf_sem, sem_names, cut - Ca, cut, **flags, origin_offset=int(a['offset'])), provenance='Arm-A %s backtest, origin -%d h, context %d h' % ('partial alternative' if a.get('partial') else 'auxiliary', a['offset'], Ca)))
    state['armB'] = False
    if use_semantic:
        armb_aux = [a for a in state.get('aux', []) if not a.get('partial') and a['offset'] == 168]
        if armb_aux:
            state['armB'] = True
            cases.append(dict(mk(th[:, L - C:L], pf_plain, plain_names, L - C, L, alternative=True, arm_b_primary=True), provenance='Primary arm B: same reconstructed history, official AH/RH/T covariates only'))
            a = armb_aux[0]
            cut = a['cut']
            Ca = int(min(cut, maxc))
            cases.append(dict(mk(np.asarray(a['hist'], float)[:, cut - Ca:cut], pf_plain, plain_names, cut - Ca, cut, auxiliary=True, arm_b_aux=True, origin_offset=int(a['offset'])), provenance='Arm-B auxiliary backtest at -168 h: same origin as the arm-A one, plain covariates, for the same-origin arm comparison'))
    return (cases, state)

def postprocess(outputs, cases, state):
    N, H = (state['N'], state['H'])
    point_a = None
    quant_a = None
    point_b = None
    quant_b = None
    tail_covs = []
    eA = eB = None
    eN = {}
    aux_iter = iter(state.get('aux', []))
    for output, case in zip(outputs, cases):
        pt = np.asarray(output['point'], float)
        qt = np.asarray(output['quantiles'], float)
        if case.get('arm_b_primary'):
            point_b, quant_b = (pt, qt)
        elif case.get('arm_b_aux'):
            a = [x for x in state.get('aux', []) if not x.get('partial') and x['offset'] == 168][0]
            eB = bt.pinball_by_target(a['truth'], a['mask'], qt)
            c9 = bt.upper_tail_coverage(a['truth'], a['mask'], qt)
            if np.isfinite(c9):
                tail_covs.append(c9)
        elif case.get('auxiliary'):
            a = next(aux_iter, None)
            if a is None:
                continue
            try:
                eN[a['offset']] = bt.pinball_by_target(a['truth'], a['mask'], qt)
                if a['offset'] == 168:
                    eA = eN[168]
                c9 = bt.upper_tail_coverage(a['truth'], a['mask'], qt)
                if np.isfinite(c9):
                    tail_covs.append(c9)
            except Exception:
                pass
        else:
            point_a, quant_a = (pt, qt)
    if point_a is None:
        raise RuntimeError('no primary case output')
    wA = 1.0
    arm_detail = None
    if point_b is not None and eA is not None and (eB is not None):
        ok = np.isfinite(eA) & np.isfinite(eB) & (eA + eB > 0)
        if ok.any():
            per = eB[ok] / (eA[ok] + eB[ok])
            wA = float(np.clip(np.mean(per), ARM_CLIP[0], ARM_CLIP[1]))
            arm_detail = {'eA_by_target': np.round(eA, 4).tolist(), 'eB_by_target': np.round(eB, 4).tolist(), 'per_target_wA': np.round(eB / np.where(eA + eB > 0, eA + eB, np.nan), 3).tolist()}
    if point_b is not None and wA < 1.0:
        la, lb = (np.log(np.clip(point_a, 0.001, None)), np.log(np.clip(point_b, 0.001, None)))
        point = np.exp(wA * la + (1 - wA) * lb)
        qa = np.log(np.clip(np.sort(quant_a, axis=2), 0.001, None))
        qb = np.log(np.clip(np.sort(quant_b, axis=2), 0.001, None))
        quantiles = np.exp(wA * qa + (1 - wA) * qb)
    else:
        point, quantiles = (point_a, quant_a)
    spec = state.get('spec')
    spec_detail = None
    if spec is not None:
        import pandas as pd
        w = np.zeros(N)
        factor = np.ones((N, H))
        rows = {}
        for i in range(N):
            if i not in spec['point']:
                continue
            margins = []
            wsum = 0.0
            for off, es in spec['eS'].get(i, {}).items():
                en = eN.get(off)
                if en is None or not np.isfinite(en[i]) or en[i] + es <= 0:
                    continue
                rw = AUX_RECENCY.get(off, 0.3)
                margins.append(rw * (en[i] - es) / (en[i] + es))
                wsum += rw
            if not margins or wsum <= 0:
                continue
            margin = sum(margins) / wsum
            w[i] = float(np.clip(2.0 * margin, 0.0, SPEC_W_MAX)) * spec['support']
            if w[i] <= 0:
                continue
            lr = np.log(np.clip(spec['point'][i], 0.001, None) / np.clip(point[i], 0.001, None))
            lr = pd.Series(lr).rolling(24, min_periods=1, center=True).mean().values
            factor[i] = np.exp(w[i] * np.clip(lr, -SPEC_LR_CAP, SPEC_LR_CAP))
            rows[state['names'][i]] = {'weight': round(float(w[i]), 3), 'backtest_margin': round(float(margin), 3), 'mean_factor': round(float(np.mean(factor[i])), 3), 'n_train': int(spec['n_train'].get(i, 0))}
        point = point * factor
        quantiles = quantiles * factor[:, :, None]
        quantiles = np.sort(quantiles, axis=2)
        spec_detail = {'support_fraction_T': round(float(spec['support']), 3), 'per_target': rows, 'native_aux_pinball': {int(o): np.round(v, 4).tolist() for o, v in eN.items()}, 'specialist_aux_pinball': {state['names'][i]: {int(o): round(v, 4) for o, v in es.items()} for i, es in spec['eS'].items()}}
    tail_cov = float(np.mean(tail_covs)) if tail_covs else np.nan
    gamma_applied = 1.0
    if np.isfinite(tail_cov) and tail_cov >= TAIL_COV_GATE:
        gamma_applied = TAIL_GAMMA
        med = quantiles[:, :, 4:5]
        upper = np.clip(med, 1e-09, None) * (np.clip(quantiles[:, :, 5:], 1e-09, None) / np.clip(med, 1e-09, None)) ** gamma_applied
        quantiles = np.sort(np.concatenate([quantiles[:, :, :4], med, upper], axis=2), axis=2)
    return {'point': point, 'quantiles': quantiles, 'components': {'mechanism': 'dual-arm covariate router + same-origin-routed weather/calendar GBM specialist (level calibration removed)', 'arm_A_weight_applied': wA, 'arm_router_backtest_pinball': arm_detail, 'upper_tail_gamma_applied': gamma_applied, 'backtest_q90_coverage_pooled': None if not np.isfinite(tail_cov) else tail_cov, 'specialist': spec_detail, 'specialist_error': state.get('spec_error'), 'calibration_from_this_origins_own_past_only': True, 'semantic_known_covariates_used': state.get('semantic_covariates', [])}}

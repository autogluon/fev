import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import retail_expert as rx
try:
    from scipy.stats import norm
    ZQ = norm.ppf(np.arange(1, 10) / 10.0)
except Exception:
    ZQ = np.array([-1.2816, -0.8416, -0.5244, -0.2533, 0.0, 0.2533, 0.5244, 0.8416, 1.2816])
BLEND_W = 0.35
ROUTER_ENABLED = False
ROUTER_ACTIONS = (0.3, 0.5, 0.7)
ROUTER_PENALTY = 0.015
AUX_MIN_CONTEXT = 64
EMPQ_N0 = 60.0
EMPQ_N1 = 150.0
EMPQ_CAP = 0.9
SPREAD_MIX = 0.7
SPREAD_SCALE = 1.0
HORIZON_GROWTH = 1.0
EMPIRICAL_Q = 0.0
PRIOR_UNCERTAINTY = 0.0
NORMALIZE = True
POINT_SHIFT = 0.0
EMP_MEDIAN_SHIFT = 0.0
CAL_PRIOR_SCALE = 1.0
ANNUAL_ALPHA = 0.0

def _calendar_parts(timestamps):
    ts = np.asarray(timestamps)
    try:
        import pandas as pd
        idx = pd.to_datetime(ts)
        months = np.asarray(idx.month, int)
        days = np.asarray(idx.day, int)
        dom_end = np.asarray(idx.days_in_month, int)
    except Exception:
        months, days, dom_end = ([], [], [])
        for s in ts:
            s = str(s)
            months.append(int(s[5:7]))
            days.append(int(s[8:10]))
            dom_end.append([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][int(s[5:7]) - 1])
        months = np.array(months)
        days = np.array(days)
        dom_end = np.array(dom_end)
    return (months, days, dom_end)

def preprocess(view, card):
    N = len(view['target_ids'])
    H = int(view['horizon'])
    L = int(view['cutoff_index'])
    known_names = list(view['known_names'])
    KF = np.asarray(view['known_features'], float) if len(view['known_names']) else np.zeros((0, L + H))
    dyn = {n: KF[j] for j, n in enumerate(known_names)}
    months, days, dom_end = _calendar_parts(view['timestamps'])
    state = {'N': N, 'H': H, 'L': L, 'dyn': dyn, 'months': months, 'days': days, 'dom_end': dom_end, 'has_open': 'Open' in dyn}
    return (view, state)

def _build_expert(dyn, state, hist_d, obs_d, Lfit, Hout):
    fit = rx.fit_item(dyn, state['months'], state['days'], state['dom_end'], hist_d[:Lfit], Lfit, observed=obs_d[:Lfit])
    if not fit['ok']:
        return None
    cal = fit['cal_series']
    train = fit['train'][:Lfit] if 'train' in fit else dyn['Open'][:Lfit] > 0.5
    norm = rx.deseasonalise(hist_d[:Lfit], fit['seasonal'], train, Lfit) if NORMALIZE else None
    idx = np.arange(Lfit)
    neff = float((0.5 ** ((Lfit - 1 - idx) / rx.HALFLIFE))[train].sum())
    return {'point': fit['fit'][Lfit:Lfit + Hout], 'sigma': fit['sigma'], 'norm': norm, 'seasonal_future': fit['seasonal'][Lfit:Lfit + Hout], 'emp_q': fit['emp_q'], 'cal_support': fit['cal_support'], 'cal_future': cal[Lfit:Lfit + Hout], 'neff': neff, 'cal_ref': rx.calendar_reference(cal, Lfit)}

def engineer(prepared, card, state):
    view = prepared
    L, H, N = (state['L'], state['H'], state['N'])
    dyn = state['dyn']
    hist = np.asarray(view['target_history'], float)
    obs = np.asarray(view.get('target_observed', np.ones_like(hist, bool)), bool)
    experts = []
    required = ('DayOfWeek', 'Open', 'Promo', 'SchoolHoliday', 'StateHoliday')
    usable = all((k in dyn for k in required))
    for d in range(N):
        exp_d = _build_expert(dyn, state, hist[d], obs[d], L, H) if usable else None
        if exp_d is not None:
            exp_d['yoy'] = rx.annual_factor(exp_d['norm'], L, H) if ANNUAL_ALPHA > 0 else None
        experts.append(exp_d)
    state['experts'] = experts
    state['usable'] = usable
    aux = []
    if ROUTER_ENABLED and usable and all((e is not None for e in experts)):
        for off in (H, 2 * H):
            Lfit = L - off
            if Lfit < AUX_MIN_CONTEXT or len(aux) >= 2:
                continue
            aux_experts = [_build_expert(dyn, state, hist[d], obs[d], Lfit, H) for d in range(N)]
            if any((e is None for e in aux_experts)):
                continue
            truth = hist[:, Lfit:Lfit + H]
            mature = obs[:, Lfit:Lfit + H] & np.isfinite(truth)
            if not mature.any():
                continue
            aux.append({'offset': off, 'Lfit': Lfit, 'experts': aux_experts, 'truth': truth, 'mature': mature})
    state['aux'] = aux
    return (prepared, state)

def select_context(variables, card, state):
    C = min(int(variables['cutoff_index']), int(card['limits']['max_context']))
    H = int(variables['horizon'])
    po = np.asarray(variables['past_features'], float) if len(variables['past_names']) else np.zeros((0, 0))
    pf = np.asarray(variables['known_features'], float) if len(variables['known_names']) else np.zeros((0, 0))
    hist = np.asarray(variables['target_history'], float)
    experts = state.get('experts') or [None] * state['N']
    norm_ok = NORMALIZE and all((e is not None and e.get('norm') is not None for e in experts))
    state['normalised'] = bool(norm_ok)
    if norm_ok:
        targets = np.stack([e['norm'] for e in experts])[:, -C:]
        if not np.isfinite(targets).all():
            norm_ok = False
            state['normalised'] = False
        else:
            cases = [{'target_indices': list(range(state['N'])), 'targets': targets, 'past_only': None, 'known_future': None, 'past_names': [], 'known_names': [], 'provenance': 'Reversibly de-seasonalised level series (calendar/promo/closure structure removed from the item own past, closed days interpolated)'}]
            kept_aux = []
            for a in state.get('aux', []):
                Ca = min(a['Lfit'], int(card['limits']['max_context']))
                tg = np.stack([e['norm'] if e['norm'] is not None else np.full(a['Lfit'], np.nan) for e in a['experts']])[:, -Ca:]
                if not np.isfinite(tg).all():
                    continue
                cases.append({'target_indices': list(range(state['N'])), 'targets': tg, 'past_only': None, 'known_future': None, 'past_names': [], 'known_names': [], 'auxiliary': True, 'origin_offset': a['offset'], 'provenance': 'Own-past native backtest at origin L-%d on the same de-seasonalised representation; used to choose the per-item expert/native blend weight' % a['offset']})
                kept_aux.append(a)
            state['aux'] = kept_aux
            return (cases, state)
    state['aux'] = []
    return ([{'target_indices': list(range(state['N'])), 'targets': hist[:, -C:], 'past_only': po[:, -C:] if po.size else None, 'known_future': pf[:, -(C + H):] if pf.size else None, 'past_names': variables['past_names'], 'known_names': variables['known_names'], 'provenance': 'Reference inputs: all targets/covariates, full available context'}], state)

def _empq_share(exp_d):
    neff = float(exp_d.get('neff', 0.0))
    return float(np.clip(EMPQ_CAP * (neff - EMPQ_N0) / max(EMPQ_N1 - EMPQ_N0, 1e-09), 0.0, EMPQ_CAP))

def _shape(exp_d):
    eq = _empq_share(exp_d)
    return (1.0 - eq) * ZQ[None, :] * exp_d['sigma'] + eq * exp_d['emp_q'][None, :]

def _blend_arm(nat_p_raw, nat_q_raw, exp_d, w, normalised):
    nat_p = np.maximum(nat_p_raw, 1.0)
    nat_q = np.maximum(nat_q_raw, 1.0)
    rel = np.log(nat_q / nat_p[:, None])
    if normalised:
        nat_adj = nat_p * np.maximum(exp_d['seasonal_future'], 1e-06)
    else:
        nat_adj = nat_p * np.exp(np.clip(exp_d['cal_future'] - exp_d['cal_ref'], -1.5, 1.5))
    loc = np.maximum(exp_d['point'], 1.0)
    p = np.exp((1.0 - w) * np.log(nat_adj) + w * np.log(loc))
    spread = (1.0 - SPREAD_MIX) * rel + SPREAD_MIX * _shape(exp_d)
    q = np.sort(p[:, None] * np.exp(spread * SPREAD_SCALE), axis=1)
    return (p, q)

def _pinball(y, q, mask):
    e = y[:, None] - q
    loss = np.maximum(QL[None, :] * e, (QL[None, :] - 1.0) * e)
    return float(loss[mask].mean()) if mask.any() else np.nan
QL = np.arange(1, 10) / 10.0

def _route_weights(state, aux_outputs):
    N = state['N']
    actions = np.array(ROUTER_ACTIONS)
    default = int(np.argmin(np.abs(actions - BLEND_W)))
    losses = np.full((len(state['aux']), N, len(actions)), np.nan)
    for j, (a, out) in enumerate(zip(state['aux'], aux_outputs)):
        pnt = np.asarray(out['point'], float)
        qnt = np.asarray(out['quantiles'], float)
        open_slice = state['dyn']['Open'][a['Lfit']:a['Lfit'] + state['H']] if state['has_open'] else None
        for d in range(N):
            exp_d = a['experts'][d]
            y = a['truth'][d]
            mask = a['mature'][d]
            for ai, w in enumerate(actions):
                p, q = _blend_arm(pnt[d], qnt[d], exp_d, w, True)
                if open_slice is not None:
                    q = np.where(open_slice[:len(p), None] < 0.5, 0.0, q)
                losses[j, d, ai] = _pinball(y, q, mask)
    weights = np.full(N, BLEND_W)
    evidence = np.zeros(N)
    for d in range(N):
        ld = losses[:, d, :]
        ok = np.isfinite(ld).all(axis=1)
        if not ok.any():
            continue
        ld = ld[ok]
        base = np.maximum(ld[:, default:default + 1], 1e-09)
        relg = (ld - base) / base
        obj = relg.mean(axis=0) + ROUTER_PENALTY * np.abs(actions - BLEND_W) / 0.2
        if len(ld) > 1:
            obj = obj + 0.5 * relg.std(axis=0) / np.sqrt(len(ld))
        weights[d] = float(actions[int(np.argmin(obj))])
        evidence[d] = float(len(ld))
    return (weights, evidence, losses)

def postprocess(outputs, cases, state):
    N, H = (state['N'], state['H'])
    point = np.empty((N, H))
    quant = np.empty((N, H, 9))
    aux_outputs = []
    for output, case in zip(outputs, cases):
        if case.get('auxiliary'):
            aux_outputs.append(output)
            continue
        point[case['target_indices']] = np.asarray(output['point'], float)
        quant[case['target_indices']] = np.asarray(output['quantiles'], float)
    if state.get('aux') and len(aux_outputs) == len(state['aux']) and state.get('normalised'):
        weights, evidence, aux_losses = _route_weights(state, aux_outputs)
    else:
        weights = np.full(N, BLEND_W)
        evidence = np.zeros(N)
        aux_losses = None
    dyn = state['dyn']
    open_future = dyn['Open'][state['L']:] if state['has_open'] else None
    out_p = np.empty_like(point)
    out_q = np.empty_like(quant)
    for d in range(N):
        nat_p = np.maximum(point[d], 1.0)
        nat_q = np.maximum(quant[d], 1.0)
        rel = np.log(nat_q / nat_p[:, None])
        exp_d = state['experts'][d] if state.get('experts') else None
        if exp_d is None:
            p = point[d].copy()
            q = quant[d].copy()
        else:
            if state.get('normalised'):
                nat_adj = nat_p * np.maximum(exp_d['seasonal_future'], 1e-06)
            else:
                nat_adj = nat_p * np.exp(np.clip(exp_d['cal_future'] - exp_d['cal_ref'], -1.5, 1.5))
            loc = np.maximum(exp_d['point'], 1.0)
            if ANNUAL_ALPHA > 0 and exp_d.get('yoy') is not None:
                loc = loc * np.clip(exp_d['yoy'], 0.5, 2.0) ** ANNUAL_ALPHA
            w_d = float(weights[d])
            p = np.exp((1.0 - w_d) * np.log(nat_adj) + w_d * np.log(loc))
            H_ = len(p)
            grow = np.sqrt(1.0 + HORIZON_GROWTH * np.arange(H_) / max(H_ - 1, 1))[:, None]
            shape = _shape(exp_d)
            if PRIOR_UNCERTAINTY > 0 and exp_d['cal_support'] < 5.0:
                extra = PRIOR_UNCERTAINTY * np.abs(exp_d['cal_future'] - exp_d['cal_ref'])[:, None]
                shape = np.sign(ZQ)[None, :] * np.sqrt(shape ** 2 + (extra * np.abs(ZQ)[None, :]) ** 2)
            expert_spread = shape * grow
            spread = (1.0 - SPREAD_MIX) * rel + SPREAD_MIX * expert_spread
            if CAL_PRIOR_SCALE != 1.0 and exp_d['cal_support'] < 5.0:
                extra = (CAL_PRIOR_SCALE - 1.0) * (exp_d['cal_future'] - exp_d['cal_ref'])
                p = p * np.exp(np.clip(extra, -1.0, 1.0))
            shift = POINT_SHIFT + EMP_MEDIAN_SHIFT * float(np.clip(exp_d['emp_q'][4], -0.1, 0.1))
            if shift:
                p = p * np.exp(shift)
            q = p[:, None] * np.exp(spread * SPREAD_SCALE)
            q = np.sort(q, axis=1)
        if open_future is not None:
            closed = open_future[:H] < 0.5
            p = np.where(closed, 0.0, p)
            q = np.where(closed[:, None], 0.0, q)
        fallback = np.maximum(point[d], 0.0)
        p = np.where(np.isfinite(p), p, fallback)
        q = np.where(np.isfinite(q), q, np.maximum(quant[d], 0.0))
        out_p[d] = np.maximum(p, 0.0)
        out_q[d] = np.sort(np.maximum(q, 0.0), axis=1)
    components = {'router_enabled': ROUTER_ENABLED, 'blend_weight_by_target': [float(w) for w in weights], 'aux_origin_offsets': [a['offset'] for a in state.get('aux', [])], 'aux_origins_used_by_target': [float(e) for e in evidence], 'actual_native_backtest_losses': None if aux_losses is None else np.round(aux_losses, 4).tolist(), 'empirical_quantile_share_by_target': [None if e is None else round(_empq_share(e), 3) for e in state.get('experts') or []], 'effective_open_day_weight_by_target': [None if e is None else round(float(e.get('neff', 0.0)), 1) for e in state.get('experts') or []], 'normalised_context': bool(state.get('normalised'))}
    return {'point': out_p, 'quantiles': out_q, 'components': components}

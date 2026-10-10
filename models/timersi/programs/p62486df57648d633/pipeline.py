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
BLEND_W = 0.5
SPREAD_MIX = 0.6
SPREAD_SCALE = 1.0
HORIZON_GROWTH = 0.0
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
        if not usable:
            experts.append(None)
            continue
        fit = rx.fit_item(dyn, state['months'], state['days'], state['dom_end'], hist[d], L, observed=obs[d])
        if not fit['ok']:
            experts.append(None)
            continue
        cal = fit['cal_series']
        usable = fit['train'][:L] if 'train' in fit else dyn['Open'][:L] > 0.5
        norm = rx.deseasonalise(hist[d], fit['seasonal'], usable, L) if NORMALIZE else None
        yoy = rx.annual_factor(norm, L, state['H']) if ANNUAL_ALPHA > 0 else None
        experts.append({'point': fit['fit'][L:], 'sigma': fit['sigma'], 'norm': norm, 'yoy': yoy, 'seasonal_future': fit['seasonal'][L:], 'emp_q': fit['emp_q'], 'cal_support': fit['cal_support'], 'cal_future': cal[L:], 'cal_ref': rx.calendar_reference(cal, L)})
    state['experts'] = experts
    state['usable'] = usable
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
        return ([{'target_indices': list(range(state['N'])), 'targets': targets, 'past_only': None, 'known_future': None, 'past_names': [], 'known_names': [], 'provenance': 'Reversibly de-seasonalised level series (calendar/promo/closure structure removed from the item own past, closed days interpolated)'}], state)
    return ([{'target_indices': list(range(state['N'])), 'targets': hist[:, -C:], 'past_only': po[:, -C:] if po.size else None, 'known_future': pf[:, -(C + H):] if pf.size else None, 'past_names': variables['past_names'], 'known_names': variables['known_names'], 'provenance': 'Reference inputs: all targets/covariates, full available context'}], state)

def postprocess(outputs, cases, state):
    N, H = (state['N'], state['H'])
    point = np.empty((N, H))
    quant = np.empty((N, H, 9))
    for output, case in zip(outputs, cases):
        point[case['target_indices']] = np.asarray(output['point'], float)
        quant[case['target_indices']] = np.asarray(output['quantiles'], float)
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
            p = np.exp((1.0 - BLEND_W) * np.log(nat_adj) + BLEND_W * np.log(loc))
            H_ = len(p)
            grow = np.sqrt(1.0 + HORIZON_GROWTH * np.arange(H_) / max(H_ - 1, 1))[:, None]
            shape = (1.0 - EMPIRICAL_Q) * ZQ[None, :] * exp_d['sigma'] + EMPIRICAL_Q * exp_d['emp_q'][None, :]
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
    return {'point': out_p, 'quantiles': out_q}

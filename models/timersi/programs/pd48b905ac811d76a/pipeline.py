import numpy as np
from calendar_utils import parse_days, holiday_tier_vector, centred_ratio_profile
from calibrate import multiply_distribution, rescale_width, horizon_width_factor
PROFILE_LAMBDA = 0.5
PROFILE_CAP = 0.06
HOLIDAY_SHRINK = 0.6
WIDTH_S0 = 0.85
WIDTH_BETA = 0.5
DESEASONALISE_INPUT = False
HOLIDAY_CLEAN_INPUT = True
MIN_LEN_FOR_DESEAS = 21

def _daily_weekly_task(view, card):
    return str(card.get('frequency', '')).upper().startswith('D') and int(card.get('seasonality', 0)) == 7

def preprocess(view, card):
    state = {'N': len(view['target_ids']), 'H': int(view['horizon']), 'L': int(view['cutoff_index'])}
    state['enabled'] = bool(_daily_weekly_task(view, card))
    if state['enabled']:
        try:
            days = parse_days(view['timestamps'])
        except Exception:
            state['enabled'] = False
            return (view, state)
        L, H = (state['L'], state['H'])
        dow = np.array([d.weekday() for d in days])
        tier = holiday_tier_vector(days)
        state['dow_past'] = dow[:L]
        state['dow_fut'] = dow[L:L + H]
        state['tier_past'] = tier[:L]
        state['tier_fut'] = tier[L:L + H]
    return (view, state)

def engineer(prepared, card, state):
    if not state['enabled']:
        return (prepared, state)
    hist = np.asarray(prepared['target_history'], dtype=float)
    obs = np.asarray(prepared['target_observed'])
    excl_base = state['tier_past'] > 0.0
    profiles, valid = ([], [])
    for i in range(state['N']):
        y = hist[i]
        bad = excl_base | ~obs[i] | ~np.isfinite(y) | (y <= 0)
        prof, ok = centred_ratio_profile(y, state['dow_past'], bad)
        profiles.append(prof)
        valid.append(ok)
    state['past_profile'] = np.array(profiles)
    state['profile_ok'] = np.array(valid)
    deseas_flag = np.zeros(state['N'], dtype=bool)
    clean_flag = np.zeros(state['N'], dtype=bool)
    clean = hist.copy()
    if (DESEASONALISE_INPUT or HOLIDAY_CLEAN_INPUT) and state['L'] >= MIN_LEN_FOR_DESEAS:
        idx = np.arange(state['L'])
        for i in range(state['N']):
            if not valid[i]:
                continue
            prof = profiles[i]
            d = hist[i] / prof[state['dow_past']]
            bad = excl_base | ~obs[i] | ~np.isfinite(d) | (d <= 0)
            if bad.all() or (~bad).sum() < MIN_LEN_FOR_DESEAS - 7:
                continue
            for t in np.where(bad)[0]:
                w = (np.abs(idx - t) <= 5) & ~bad
                d[t] = np.median(d[w]) if w.any() else np.median(d[~bad])
            if DESEASONALISE_INPUT:
                clean[i] = d
                deseas_flag[i] = True
            else:
                clean[i] = d * prof[state['dow_past']]
                clean_flag[i] = True
    if deseas_flag.any() or clean_flag.any():
        variables = dict(prepared)
        variables['target_history'] = clean
        state['deseas'] = deseas_flag
        state['cleaned'] = clean_flag
        return (variables, state)
    state['deseas'] = deseas_flag
    state['cleaned'] = clean_flag
    return (prepared, state)

def select_context(variables, card, state):
    C = min(variables['cutoff_index'], card['limits']['max_context'])
    H = variables['horizon']
    po = variables['past_features']
    pf = variables['known_features']
    return ([{'target_indices': list(range(state['N'])), 'targets': variables['target_history'][:, -C:], 'past_only': po[:, -C:] if len(po) else None, 'known_future': pf[:, -(C + H):] if len(pf) else None, 'past_names': variables['past_names'], 'known_names': variables['known_names'], 'provenance': 'Official reference inputs, all targets/covariates, full context'}], state)

def _reseasonalise(i, state):
    if not state['deseas'][i]:
        return np.ones(state['H'])
    return state['past_profile'][i][state['dow_fut']]

def _corrections(fc_point, i, state):
    H = state['H']
    mult = np.ones(H)
    dowf = state['dow_fut']
    prof = state['past_profile'][i]
    ok = state['profile_ok'][i]
    if ok and H >= 9:
        fprof, fok = centred_ratio_profile(fc_point, dowf, np.zeros(H, dtype=bool))
        if fok:
            with np.errstate(invalid='ignore', divide='ignore'):
                ratio = prof / fprof
            g = np.clip(ratio[dowf] ** PROFILE_LAMBDA, 1.0 - PROFILE_CAP, 1.0 + PROFILE_CAP)
            if np.all(np.isfinite(g)) and np.all(g > 0):
                g = g / float(np.exp(np.mean(np.log(g))))
                mult *= g
    if ok:
        gap = prof[5] / np.maximum(prof[dowf], 1e-09) - 1.0
        gap = np.clip(gap, 0.0, 0.35)
        weekday = dowf < 5
        mult *= 1.0 + HOLIDAY_SHRINK * state['tier_fut'] * gap * weekday
    return mult

def postprocess(outputs, cases, state):
    N, H = (state['N'], state['H'])
    point = np.empty((N, H))
    quantiles = np.empty((N, H, 9))
    for output, case in zip(outputs, cases):
        p = np.asarray(output['point'], dtype=float)
        q = np.asarray(output['quantiles'], dtype=float)
        for row, idx in enumerate(case['target_indices']):
            pi, qi = (p[row], q[row])
            if state['enabled']:
                fac = horizon_width_factor(qi, WIDTH_S0, WIDTH_BETA)
                back = _reseasonalise(idx, state)
                pi, qi = multiply_distribution(pi, qi, back)
                mult = _corrections(pi, idx, state)
                pi, qi = multiply_distribution(pi, qi, mult)
                qi = rescale_width(pi, qi, fac)
            qi = np.sort(qi, axis=-1)
            point[idx] = pi
            quantiles[idx] = qi
    return {'point': point, 'quantiles': quantiles}

import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calendar_model as cm
import closure as clo
import context as ctx
NORMALIZE = True
KNOWN_MODE = 'semantic'
BLEND_W = 0.15
SHAPE_A = 0.2
MAX_ADJ = 0.15

def preprocess(view, card):
    state = {'N': len(view['target_ids']), 'H': view['horizon'], 'L': view['cutoff_index']}
    hist = np.array(view['target_history'], float)
    raw_tail = hist[:, -28:].copy()
    dates = cm._dates(view['timestamps'])
    dow = cm._dow(dates)
    starts, cleaned = ([], hist.copy())
    for d in range(hist.shape[0]):
        y = hist[d, :view['cutoff_index']]
        s = ctx.active_start(y)
        filled, _ = ctx.impute_isolated_zeros(y[s:], dow[s:view['cutoff_index']])
        cleaned[d, s:view['cutoff_index']] = filled
        starts.append(s)
    state['starts'] = starts
    state['raw_tail'] = raw_tail
    state['start'] = int(min(starts))
    state['raw_history'] = hist
    prepared = dict(view)
    prepared['target_history'] = cleaned
    return (prepared, state)

def engineer(prepared, card, state):
    known = np.array(prepared['known_features'], float) if len(prepared['known_features']) else None
    hol = None
    if known is not None and known.size:
        names = list(prepared['known_names'])
        if 'holiday' in names:
            hol = known[names.index('holiday')]
    models = []
    for d in range(state['N']):
        try:
            models.append(cm.fit(prepared['target_history'][d], prepared['timestamps'], state['L'], state['H'], holiday=hol, start=state['starts'][d]))
        except Exception:
            models.append(None)
    state['models'] = models
    state['holiday_row'] = hol
    factors = np.ones((state['N'], state['L'] + state['H']))
    if NORMALIZE:
        dates = cm._dates(prepared['timestamps'])
        dow_all, dom_all = (cm._dow(dates), cm._ymd(dates)[2])
        hol_all = np.asarray(hol, float) if hol is not None else np.zeros(len(dates))
        for d in range(state['N']):
            model = models[d]
            if model is None:
                continue
            eff = np.array([model['dowf'].get(int(a), 0.0) + model['domf'].get(int(b), 0.0) for a, b in zip(dow_all, dom_all)])
            factors[d] = np.exp(np.clip(eff, -0.25, 0.25))
    state['factors'] = factors
    if hol is not None and KNOWN_MODE == 'semantic':
        flag = (np.asarray(hol, float) != 0).astype(float)
        effect = np.zeros_like(flag)
        for d in range(state['N']):
            model = models[d]
            if model is None:
                continue
            table = model['holf']
            fallback = float(np.median(list(table.values()))) if table else 0.0
            for t, code in enumerate(np.asarray(hol, float)):
                if code != 0:
                    effect[t] = table.get(float(code), fallback)
            break
        state['known_semantic'] = np.vstack([flag, effect])
        state['known_semantic_names'] = np.array(['holiday_flag', 'holiday_effect'])
    return (prepared, state)

def select_context(variables, card, state):
    L, H = (state['L'], state['H'])
    C = min(L, card['limits']['max_context'])
    po = np.array(variables['past_features'], float) if len(variables['past_features']) else None
    pf = np.array(variables['known_features'], float) if len(variables['known_features']) else None
    known_names = variables['known_names']
    if state.get('known_semantic') is not None:
        pf = state['known_semantic']
        known_names = state['known_semantic_names']
    case = {'target_indices': list(range(state['N'])), 'targets': np.asarray(state['raw_history'], float)[:, L - C:L] / state['factors'][:, L - C:L], 'past_only': po[:, L - C:L] if po is not None and po.size else None, 'known_future': pf[:, L - C:L + H] if pf is not None and pf.size else None, 'past_names': variables['past_names'], 'known_names': known_names, 'provenance': 'full official context, own-past calendar-normalised target, holiday hash re-encoded as flag + own-past log effect'}
    state['C'] = C
    return ([case], state)

def _closed_or_broken(raw_tail):
    tail = np.asarray(raw_tail, float)
    return (tail <= 0).mean() > 0.1

def postprocess(outputs, cases, state):
    N, H = (state['N'], state['H'])
    point = np.empty((N, H))
    quantiles = np.empty((N, H, 9))
    for output, case in zip(outputs, cases):
        point[case['target_indices']] = np.asarray(output['point'], float)
        quantiles[case['target_indices']] = np.asarray(output['quantiles'], float)
    fut = state['factors'][:, state['L']:state['L'] + H]
    point = point * fut
    quantiles = quantiles * fut[:, :, None]
    for d in range(N):
        model = state['models'][d]
        base = point[d]
        if model is None or _closed_or_broken(state['raw_tail'][d]):
            continue
        local = cm.predict(model)
        if not np.isfinite(local).all() or (local <= 0).any() or (base <= 0).any():
            continue
        target = (1.0 - BLEND_W) * base + BLEND_W * local
        if SHAPE_A > 0:
            shape = local / local.mean() / (base / base.mean())
            shape = np.power(np.clip(shape, 0.5, 2.0), SHAPE_A)
            shape = shape / (np.mean(shape * base) / base.mean())
            target = target * shape
        ratio = np.clip(target / base, 1.0 - MAX_ADJ, 1.0 + MAX_ADJ)
        if not np.isfinite(ratio).all():
            continue
        point[d] = base * ratio
        quantiles[d] = quantiles[d] * ratio[:, None]
    hol = state.get('holiday_row')
    for d in range(N):
        raw = np.asarray(state['raw_history'], float)[d, :state['L']]
        risk = clo.trailing_zero_risk(raw, hol[:state['L']] if hol is not None else None)
        if risk > 0:
            point[d], quantiles[d] = clo.apply_mixture(point[d], quantiles[d], risk)
    quantiles = np.sort(quantiles, axis=2)
    return {'point': point, 'quantiles': quantiles}

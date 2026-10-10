import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mechanisms import QL, activity_plan, anchor_level, launch_index, level_trend_correction, mixture_quantiles, paused_forecast, robust_sigma, useful_known_channel, useful_past_channel
LAM_C0 = 30.0
LAM_CAP = 0.7
KAPPA_TREND = 0.5
KANCH = 8
W_PAUSED = 0.6
PSEUDO = 0.5
USE_LOG = False
USE_ACTIVITY_CONTEXT = True
MIN_ACTIVE = 4
W_EMP = 0.0
EMP_LOOKBACK = 26
PRUNE_COVARIATES = True

def preprocess(view, card):
    hist = np.asarray(view['target_history'], dtype=float)
    obs = np.asarray(view['target_observed'])
    nonneg = bool(np.all(hist >= 0))
    H = int(view['horizon'])
    plans = []
    if USE_ACTIVITY_CONTEXT and nonneg and (hist.shape[0] == 1):
        plans = [activity_plan(hist[0], H, MIN_ACTIVE)]
    state = {'N': len(view['target_ids']), 'H': H, 'C': int(view['cutoff_index']), 'hist': hist, 'obs': obs, 'nonneg': nonneg, 'log': bool(USE_LOG and nonneg), 'plans': plans}
    return (view, state)

def engineer(prepared, card, state):
    return (prepared, state)

def _prune(po, pf, names_p, names_k, target, n_past):
    if not PRUNE_COVARIATES:
        return (po, pf, names_p, names_k)
    if po is not None and len(po):
        keep = [i for i in range(po.shape[0]) if useful_past_channel(po[i], target)]
        po = po[keep] if keep else None
        names_p = [names_p[i] for i in keep] if keep else []
    if pf is not None and len(pf):
        keep = [i for i in range(pf.shape[0]) if useful_known_channel(pf[i], n_past)]
        pf = pf[keep] if keep else None
        names_k = [names_k[i] for i in keep] if keep else []
    return (po, pf, names_p, names_k)

def select_context(variables, card, state):
    C = min(variables['cutoff_index'], card['limits']['max_context'])
    H = variables['horizon']
    po = variables['past_features']
    pf = variables['known_features']
    plans = state.get('plans') or []
    plan = plans[0] if len(plans) == 1 and plans[0] is not None else None
    if plan is not None and (not plan['trivial']):
        idx = plan['idx'][-C:]
        state['ctx_idx'] = idx
        po_s = np.asarray(variables['past_features'])[:, idx] if len(po) else None
        if len(pf):
            L = int(variables['cutoff_index'])
            pf_a = np.asarray(variables['known_features'])
            pf_s = np.concatenate([pf_a[:, idx], pf_a[:, L:L + H]], axis=1)
        else:
            pf_s = None
        tgt_s = np.asarray(variables['target_history'], dtype=float)[:, idx]
        po_s, pf_s, npn, nkn = _prune(po_s, pf_s, list(variables['past_names']), list(variables['known_names']), tgt_s[0], len(idx))
        return ([{'target_indices': list(range(state['N'])), 'targets': tgt_s, 'past_only': po_s, 'known_future': pf_s, 'past_names': npn, 'known_names': nkn, 'provenance': 'activity-aware context: pre-launch padding and outage weeks removed, covariates sub-selected on the same calendar indices'}], state)
    tgt = np.asarray(variables['target_history'], dtype=float)[:, -C:]
    if state['log']:
        tgt = np.log1p(np.maximum(tgt, 0.0))
        prov = 'reference context, log1p reversible target normalisation'
    else:
        prov = 'reference context, raw units (history has negatives)'
    po_s = np.asarray(po)[:, -C:] if len(po) else None
    pf_s = np.asarray(pf)[:, -(C + H):] if len(pf) else None
    po_s, pf_s, npn, nkn = _prune(po_s, pf_s, list(variables['past_names']), list(variables['known_names']), tgt[0], C)
    return ([{'target_indices': list(range(state['N'])), 'targets': tgt, 'past_only': po_s, 'known_future': pf_s, 'past_names': npn, 'known_names': nkn, 'provenance': prov + '; covariate screening'}], state)

def postprocess(outputs, cases, state):
    N, H = (state['N'], state['H'])
    point = np.empty((N, H))
    quant = np.empty((N, H, 9))
    for output, case in zip(outputs, cases):
        point[case['target_indices']] = output['point']
        quant[case['target_indices']] = output['quantiles']
    if state['log']:
        point = np.expm1(np.clip(point, -30.0, 30.0))
        quant = np.expm1(np.clip(quant, -30.0, 30.0))
    hist = state['hist']
    plans = state.get('plans') or []
    for d in range(N):
        h = hist[d]
        if launch_index(h) < 0:
            continue
        plan = plans[d] if d < len(plans) else None
        if plan is not None and (not plan['trivial']):
            act = h[plan['idx']]
            p_old = point[d]
            p_new = level_trend_correction(p_old, act, H, LAM_C0, LAM_CAP, KAPPA_TREND, KANCH)
            aq = np.sort(quant[d] + (p_new - p_old)[:, None], axis=-1)
            aq = np.maximum(aq, 0.0)
            if W_EMP > 0.0 and len(act) >= 3:
                emp = np.quantile(act[-EMP_LOOKBACK:], QL)
                aq = (1.0 - W_EMP) * aq + W_EMP * emp[None, :]
            pa = plan['p'] if h[-1] == 0 else np.ones(H)
            quant[d] = np.stack([mixture_quantiles(pa[j], aq[j]) for j in range(H)])
            point[d] = quant[d][:, 4]
            quant[d] = np.sort(quant[d], axis=-1)
            if state['nonneg']:
                quant[d] = np.maximum(quant[d], 0.0)
                point[d] = np.maximum(point[d], 0.0)
            continue
        pz = paused_forecast(h, H, PSEUDO)
        if pz is not None:
            Qn, Pn = pz
            quant[d] = (1.0 - W_PAUSED) * quant[d] + W_PAUSED * Qn
            point[d] = quant[d][:, 4]
        else:
            p_old = point[d]
            p_new = level_trend_correction(p_old, h, H, LAM_C0, LAM_CAP, KAPPA_TREND, KANCH)
            quant[d] = quant[d] + (p_new - p_old)[:, None]
            point[d] = p_new
        quant[d] = np.sort(quant[d], axis=-1)
        if state['nonneg']:
            quant[d] = np.maximum(quant[d], 0.0)
            point[d] = np.maximum(point[d], 0.0)
    return {'point': point, 'quantiles': quant}

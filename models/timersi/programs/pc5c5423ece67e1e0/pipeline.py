import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corelib
_POOL = {}
_FIT_CACHE = {}
KAPPA_NATIVE = 0.0
FALLBACK_MIN_CTX = 4

def _clean(row, observed):
    y = np.asarray(row, dtype=float).copy()
    ok = np.asarray(observed, dtype=bool) & np.isfinite(y)
    if not ok.any():
        return None
    if not ok.all():
        idx = np.arange(len(y))
        y[~ok] = np.interp(idx[~ok], idx[ok], y[ok])
    return y

def preprocess(view, card):
    hist = np.asarray(view['target_history'], dtype=float)
    obs = np.asarray(view['target_observed'], dtype=bool)
    N, L = hist.shape
    names = [str(n) for n in view['target_names']]
    clean = {}
    for d in range(N):
        y = _clean(hist[d], obs[d])
        if y is not None:
            clean[d] = y
            _POOL.setdefault(names[d], {})[int(view['item_index'])] = y
    _FIT_CACHE.clear()
    state = {'N': N, 'L': L, 'H': int(view['horizon']), 'names': names, 'item_index': int(view['item_index']), 'clean': clean}
    return (view, state)

def engineer(prepared, card, state):
    ts = np.asarray(prepared['timestamps'])
    state['n_times'] = int(len(ts))
    diag = {}
    for d, y in state['clean'].items():
        n = len(y)
        if n > 1:
            step = np.abs(np.diff(y))
            scale = float(np.mean(step)) if len(step) else 1.0
        else:
            scale = 1.0
        diag[d] = {'n': n, 'mean': float(np.mean(y)), 'sd': float(np.std(y)), 'scale': scale, 'lag1': float(np.corrcoef(y[1:], y[:-1])[0, 1]) if n > 3 and np.std(y) > 0 else 0.0}
    state['diag'] = diag
    return (prepared, state)

def select_context(variables, card, state):
    C = min(variables['cutoff_index'], card['limits']['max_context'])
    H = variables['horizon']
    po = variables['past_features']
    pf = variables['known_features']
    return ([{'target_indices': list(range(state['N'])), 'targets': variables['target_history'][:, -C:], 'past_only': po[:, -C:] if len(po) else None, 'known_future': pf[:, -(C + H):] if len(pf) else None, 'past_names': variables['past_names'], 'known_names': variables['known_names'], 'provenance': 'Official reference inputs, all targets/covariates, max_context=15360'}], state)

def _fit(name, H):
    key = (name, H, tuple(sorted(_POOL.get(name, {}).keys())))
    if key in _FIT_CACHE:
        return _FIT_CACHE[key]
    entries = sorted(_POOL.get(name, {}).items())
    if not entries:
        return None
    order = [i for i, _ in entries]
    hists = [v for _, v in entries]
    no_ar = [c for c in corelib.default_candidates() if not c[3]]
    variants, infos = ([], [])
    for ow, frac, cands in ((True, 3, None), (False, 4, None), (False, 4, no_ar)):
        try:
            variants.append(corelib.fit_pool(hists, H, origin_weight=ow, min_ctx_frac=frac, candidates=cands))
            infos.append(variants[-1][3])
        except Exception:
            continue
    if not variants:
        _FIT_CACHE[key] = None
        return None
    centres = np.mean([v[0] for v in variants], axis=0)
    sd = np.mean([v[1] for v in variants], axis=0)
    shape = np.mean([v[2] for v in variants], axis=0)
    info = dict(infos[0])
    info['ensemble'] = len(variants)
    info['alpha_eff'] = float(np.mean([i['alpha_eff'] for i in infos]))
    info['sigma'] = float(np.mean([i['sigma'] for i in infos]))
    info['pool_size'] = len(hists)
    res = ({idx: centres[p] for p, idx in enumerate(order)}, {idx: sd[p] for p, idx in enumerate(order)}, shape, info)
    _FIT_CACHE[key] = res
    return res

def postprocess(outputs, cases, state):
    N, H = (state['N'], state['H'])
    point = np.empty((N, H))
    quantiles = np.empty((N, H, 9))
    for output, case in zip(outputs, cases):
        point[case['target_indices']] = output['point']
        quantiles[case['target_indices']] = output['quantiles']
    native_q = quantiles.copy()
    components = {'engine': 'pooled-local-level-EB', 'kappa_native': KAPPA_NATIVE, 'targets': {}}
    item = state['item_index']
    for d in range(N):
        name = state['names'][d]
        if d not in state['clean'] or state['L'] < FALLBACK_MIN_CTX:
            components['targets'][name] = {'mode': 'native-fallback'}
            continue
        fit = _fit(name, H)
        if fit is None or item not in fit[0]:
            components['targets'][name] = {'mode': 'native-fallback'}
            continue
        centres, sd_all, shape, info = fit
        sd = np.asarray(sd_all[item], dtype=float)
        c = np.asarray(centres[item], dtype=float).copy()
        if KAPPA_NATIVE > 0.0:
            med = native_q[d, :, 4]
            c = c + KAPPA_NATIVE * (med - float(np.mean(med)))
        q = c[:, None] + sd[:, None] * shape[None, :]
        q = np.sort(q, axis=-1)
        if not np.all(np.isfinite(q)):
            components['targets'][name] = {'mode': 'native-fallback-nonfinite'}
            continue
        quantiles[d] = q
        point[d] = q[:, 4]
        components['targets'][name] = {'mode': 'pooled-eb', 'pool_size': info['pool_size'], 'alpha_eff': info['alpha_eff'], 'sigma': info['sigma'], 'level': float(c[0]), 'sd_h1': float(sd[0]), 'sd_hH': float(sd[-1]), 'shape': info['shape'], 'best_candidates': info['best_candidates'], 'rho': info['rho'], 'w_scale': info['w_scale'], 'scale_factor': info['scale_factor'][0] if info['scale_factor'] else 1.0, 'native_level': float(np.mean(native_q[d, :, 4]))}
    return {'point': point, 'quantiles': quantiles, 'components': components}

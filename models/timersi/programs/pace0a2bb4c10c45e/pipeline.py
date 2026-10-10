import numpy as np
import series as _series
import calib as _calib

def preprocess(view, card):
    state = {'N': len(view['target_ids']), 'H': view['horizon'], 'notes': []}
    hist = np.asarray(view['target_history'], dtype=float)
    obs = np.asarray(view['target_observed'], dtype=bool)
    new = hist.copy()
    flag = np.zeros_like(hist)
    for d in range(hist.shape[0]):
        try:
            rebuilt, gap = _series.reconstruct(hist[d], obs[d], view['timestamps'])
            if np.all(np.isfinite(rebuilt)):
                new[d] = rebuilt
                flag[d] = gap
        except Exception as exc:
            state['notes'].append('reconstruct failed: %r' % (exc,))
    state['observed_fraction'] = float(obs.mean()) if obs.size else 1.0
    state['raw_history'] = hist
    state['observed'] = obs
    state['n_observed'] = obs.sum(axis=1).astype(float)
    known = np.asarray(view['known_features'], dtype=float)
    depth = _calib.promo_depth(known, list(view['known_names'])) if len(known) else None
    L, H = (view['cutoff_index'], view['horizon'])
    state['past_depth'] = depth[:L] if depth is not None else None
    state['future_depth'] = depth[L:L + H] if depth is not None else None
    state['tail_observed'] = obs[:, -28:].mean(axis=1) if obs.shape[1] >= 28 else obs.mean(axis=1)
    prepared = dict(view)
    prepared['target_history'] = new
    prepared['imputed_flag'] = flag
    return (prepared, state)

def _prune(mat, names, keep_always=()):
    if mat is None or len(mat) == 0:
        return (mat, list(names), [])
    arr = np.asarray(mat, dtype=float)
    keep, dropped = ([], [])
    for i, nm in enumerate(names):
        row = arr[i]
        finite = row[np.isfinite(row)]
        informative = finite.size > 0 and float(np.nanmax(finite) - np.nanmin(finite)) > 1e-12
        if informative or nm in keep_always:
            keep.append(i)
        else:
            dropped.append(nm)
    if not keep:
        return (np.zeros((0, arr.shape[1])), [], dropped)
    return (arr[keep], [names[i] for i in keep], dropped)

def engineer(prepared, card, state):
    past = np.asarray(prepared['past_features'], dtype=float)
    names = list(prepared['past_names'])
    flag = prepared.get('imputed_flag')
    variables = dict(prepared)
    if flag is not None and np.any(flag > 0):
        extra = flag[:1] if flag.shape[0] >= 1 else None
        if extra is not None and np.all(np.isfinite(extra)):
            past = np.concatenate([past, extra], axis=0) if len(past) else extra
            names = names + ['imputed']
    past, names, dropped_past = _prune(past, names)
    known = np.asarray(prepared['known_features'], dtype=float)
    known, known_names, dropped_known = _prune(known, list(prepared['known_names']))
    state['dropped'] = dropped_past + dropped_known
    state['long_gap_cells'] = float(np.sum(flag)) if flag is not None else 0.0
    variables['past_features'] = past
    variables['past_names'] = names
    variables['known_features'] = known
    variables['known_names'] = known_names
    return (variables, state)

def select_context(variables, card, state):
    C = min(variables['cutoff_index'], card['limits']['max_context'])
    H = variables['horizon']
    po = np.asarray(variables['past_features'], dtype=float)
    pf = np.asarray(variables['known_features'], dtype=float)
    targets = np.asarray(variables['target_history'], dtype=float)[:, -C:]
    if not np.all(np.isfinite(targets)):
        targets = np.nan_to_num(targets, nan=float(np.nanmedian(targets)) if np.isfinite(np.nanmedian(targets)) else 0.0)
    case = {'target_indices': list(range(state['N'])), 'targets': targets, 'past_only': po[:, -C:] if len(po) else None, 'known_future': pf[:, -(C + H):] if len(pf) else None, 'past_names': variables['past_names'], 'known_names': variables['known_names'], 'provenance': 'long-gap reconstruction + zero-variance covariate pruning, max_context=15360'}
    return ([case], state)

def postprocess(outputs, cases, state):
    point = np.empty((state['N'], state['H']))
    quantiles = np.empty((state['N'], state['H'], 9))
    for output, case in zip(outputs, cases):
        point[case['target_indices']] = output['point']
        quantiles[case['target_indices']] = np.sort(np.asarray(output['quantiles'], dtype=float), axis=-1)
    for d in range(state['N']):
        p, q = (point[d].astype(float), quantiles[d].astype(float))
        try:
            p, q = _calib.promo_correction(p, q, state['raw_history'][d], state['observed'][d], state['past_depth'], state['future_depth'])
        except Exception as exc:
            state['notes'].append('promo_correction failed: %r' % (exc,))
        try:
            factor = _calib.young_trend_factor(state['raw_history'][d], state['observed'][d], state['H'])
            if factor is not None:
                p = p * factor
                q = np.sort(q * factor[:, None], axis=-1)
        except Exception as exc:
            state['notes'].append('young_trend failed: %r' % (exc,))
        try:
            spread = _calib.young_spread(state['n_observed'][d], np.atleast_1d(state['tail_observed'])[d])
            if spread > 1.0:
                q = _calib.widen(q, np.full(state['H'], spread))
        except Exception as exc:
            state['notes'].append('young_spread failed: %r' % (exc,))
        if np.all(np.isfinite(p)) and np.all(np.isfinite(q)):
            point[d], quantiles[d] = (p, np.sort(q, axis=-1))
    return {'point': point, 'quantiles': quantiles}

import numpy as np
import structure

def _names(variables):
    pn = variables.get('past_names')
    kn = variables.get('known_names')
    pn = list(pn) if pn is not None and len(pn) else []
    kn = list(kn) if kn is not None and len(kn) else []
    return (pn, kn)

def _slice_cov(variables, lo, hi, hi_known):
    po = variables.get('past_features')
    pf = variables.get('known_features')
    po = po[:, lo:hi] if po is not None and len(po) else None
    pf = pf[:, lo:hi_known] if pf is not None and len(pf) else None
    return (po, pf)

def primary(variables, state, context):
    L = int(variables['cutoff_index'])
    H = int(variables['horizon'])
    C = int(min(context, L))
    pn, kn = _names(variables)
    po, pf = _slice_cov(variables, L - C, L, L + H)
    return {'target_indices': list(range(state['N'])), 'targets': variables['target_history'][:, L - C:L], 'past_only': po, 'known_future': pf, 'past_names': pn, 'known_names': kn, 'provenance': 'primary: all %d targets, own past, context=%d' % (state['N'], C)}

def alternative_context(variables, state, context):
    case = primary(variables, state, context)
    case['alternative'] = True
    case['provenance'] = 'alternative: shorter context=%d, recent phase regime' % case['targets'].shape[1]
    return case

def alternative_compressed(variables, state, context):
    L = int(variables['cutoff_index'])
    H = int(variables['horizon'])
    C = int(min(context, L))
    Z, _ = structure.compress(variables['target_history'][:, L - C:L], state['floor'], state['pos_scale'])
    pn, kn = _names(variables)
    po, pf = _slice_cov(variables, L - C, L, L + H)
    return {'target_indices': list(range(state['N'])), 'targets': np.ascontiguousarray(Z), 'past_only': po, 'known_future': pf, 'past_names': pn, 'known_names': kn, 'alternative': True, 'provenance': 'alternative: reversible log1p((y-floor)/scale), context=%d' % C}

def auxiliary(variables, state, offset, context):
    L = int(variables['cutoff_index'])
    H = int(variables['horizon'])
    end = L - int(offset)
    C = int(min(context, end))
    if C < 64 or end + H > L:
        return None
    pn, kn = _names(variables)
    po, pf = _slice_cov(variables, end - C, end, end + H)
    return {'target_indices': list(range(state['N'])), 'targets': variables['target_history'][:, end - C:end], 'past_only': po, 'known_future': pf, 'past_names': pn, 'known_names': kn, 'auxiliary': True, 'origin_offset': int(offset), 'provenance': 'auxiliary backtest: origin L-%d, context=%d' % (offset, C)}

def alternative_group(variables, state, context, size=32):
    L = int(variables['cutoff_index'])
    H = int(variables['horizon'])
    C = int(min(context, L))
    Y = variables['target_history'][:, L - C:L]
    n = Y.shape[0]
    if n <= size:
        return None
    w = min(C, 4096)
    corr = np.nan_to_num(np.corrcoef(Y[:, -w:]))
    seed = int(np.argmax((corr > 0.8).sum(axis=1)))
    order = np.sort(np.argsort(-corr[seed])[:size])
    pn, kn = _names(variables)
    po, pf = _slice_cov(variables, L - C, L, L + H)
    return {'target_indices': [int(i) for i in order], 'targets': np.ascontiguousarray(Y[order]), 'past_only': po, 'known_future': pf, 'past_names': pn, 'known_names': kn, 'alternative': True, 'provenance': 'alternative: correlated cluster of %d targets, context=%d (grouping sensitivity measurement)' % (size, C)}

import numpy as np
try:
    from sawtooth import fit_reset_ramp, simulate, backtest_gain
except Exception:
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from sawtooth import fit_reset_ramp, simulate, backtest_gain
QS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
MIN_GAIN = 0.02
PARTICLES = 5000
SUBSTEPS = 4
JITTER = 0.05

def preprocess(view, card):
    state = {'N': len(view['target_ids']), 'H': view['horizon'], 'hist': np.asarray(view['target_history'], dtype=np.float64), 'obs': np.asarray(view['target_observed']) if view.get('target_observed') is not None else None}
    return (view, state)

def engineer(prepared, card, state):
    hist = state['hist']
    H = state['H']
    fits = {}
    diag = []
    for i in range(state['N']):
        x = hist[i]
        if state['obs'] is not None:
            m = np.asarray(state['obs'][i], dtype=bool)
            if m.size == x.size and (not m.all()):
                if m.sum() < 400:
                    continue
                x = x[m]
        try:
            f = fit_reset_ramp(x)
        except Exception:
            f = None
        if f is None:
            diag.append((i, 'nofit', 0.0, 0))
            continue
        try:
            gain, nbt = backtest_gain(x, f, H, QS, jitter=JITTER * abs(f['s']), pair=True, gjitter=0.4)
        except Exception:
            gain, nbt = (0.0, 0)
        diag.append((i, 'fit', gain, f['ncyc']))
        if nbt >= 3 and gain > MIN_GAIN:
            fits[i] = (f, x)
    state['fits'] = fits
    state['diag'] = diag
    return (prepared, state)

def select_context(variables, card, state):
    C = min(variables['cutoff_index'], card['limits']['max_context'])
    H = variables['horizon']
    po = variables['past_features']
    pf = variables['known_features']
    hist = variables['target_history']
    ramp = sorted(state.get('fits', {}).keys())
    rest = [i for i in range(state['N']) if i not in set(ramp)]
    groups = []
    for label, g in (('reset-ramp regime', ramp), ('burst/noise regime', rest)):
        if not g:
            continue
        if len(g) > 32:
            for a in range(0, len(g), 32):
                groups.append((label, g[a:a + 32]))
        else:
            groups.append((label, g))
    if not groups:
        groups = [('all targets', list(range(state['N'])))]
    cases = []
    for label, g in groups:
        cases.append({'target_indices': list(g), 'targets': hist[g][:, -C:], 'past_only': po[:, -C:] if len(po) else None, 'known_future': pf[:, -(C + H):] if len(pf) else None, 'past_names': variables['past_names'], 'known_names': variables['known_names'], 'provenance': 'Official reference inputs, max_context=15360, %s (%d variates)' % (label, len(g))})
    return (cases, state)

def postprocess(outputs, cases, state):
    D, H = (state['N'], state['H'])
    point = np.empty((D, H))
    quantiles = np.empty((D, H, 9))
    for output, case in zip(outputs, cases):
        point[case['target_indices']] = output['point']
        quantiles[case['target_indices']] = output['quantiles']
    for i, (f, x) in state.get('fits', {}).items():
        try:
            q, _, _ = simulate(x, f, H, QS, M=PARTICLES, N=SUBSTEPS, seed=7919 + i, jitter=JITTER * abs(f['s']), pair=True, gjitter=0.4)
        except Exception:
            continue
        if not np.isfinite(q).all():
            continue
        q = np.sort(q, axis=1)
        quantiles[i] = q
        point[i] = q[:, 4]
    return {'point': point, 'quantiles': quantiles}

import numpy as np
MIN_BACKTEST_CONTEXT = 16
MIN_SEASONAL_CONTEXT = 24
SEASON = 52

def live_mask(block, eps=1e-12):
    if block is None or len(block) == 0:
        return np.zeros(0, dtype=bool)
    b = np.asarray(block, dtype=float)
    return np.nanmax(b, axis=1) - np.nanmin(b, axis=1) > eps

def _subset(block, mask, names):
    if block is None or len(block) == 0 or (not mask.any()):
        return (None, [])
    return (np.asarray(block, dtype=float)[mask], [n for n, m in zip(names, mask) if m])

def _iso_weeks(stamps):
    out = []
    for s in np.asarray(stamps).ravel():
        try:
            out.append(int(np.datetime64(str(s)[:10], 'D').astype('O').isocalendar()[1]))
        except Exception:
            out.append(0)
    return np.asarray(out, dtype=int)

def choose_offsets(L, H, stamps, n_aux=2, min_ctx=MIN_BACKTEST_CONTEXT):
    chosen, roles = ([], [])
    if n_aux <= 0 or L - H < min_ctx:
        return []
    chosen.append(H)
    roles.append('recent')
    weeks = _iso_weeks(stamps)
    fut = set(weeks[L:L + H].tolist()) if len(weeks) >= L + H else set()
    best, best_key = (None, None)
    for o in range(H + 2, L - MIN_SEASONAL_CONTEXT + 1):
        e = L - o
        if e < MIN_SEASONAL_CONTEXT or e + H > L:
            continue
        aw = set(weeks[e:e + H].tolist())
        overlap = len(fut & aw) if fut else 0
        if overlap < max(1, H // 2):
            continue
        key = (overlap, -abs(o - SEASON), e)
        if best_key is None or key > best_key:
            best_key, best = (key, o)
    if len(chosen) < n_aux and best is not None:
        chosen.append(best)
        roles.append('seasonal')
    k = 2
    while len(chosen) < n_aux and H * (k + 1) <= L - min_ctx:
        o = H * (k + 1)
        if o not in chosen:
            chosen.append(o)
            roles.append('spread')
        k += 1
    return list(zip(chosen, roles))

def build_cases(variables, state, max_context, prune=False, n_aux=0):
    L = int(variables['cutoff_index'])
    H = int(variables['horizon'])
    N = state['N']
    Y = np.asarray(variables['target_history'], dtype=float)
    P = np.asarray(variables['past_features'], dtype=float) if len(variables['past_features']) else np.zeros((0, L))
    K = np.asarray(variables['known_features'], dtype=float) if len(variables['known_features']) else np.zeros((0, L + H))
    pn = list(variables['past_names'])
    kn = list(variables['known_names'])
    stamps = list(variables.get('timestamps', []))
    C = min(L, max_context)
    pm = live_mask(P[:, L - C:]) if prune else np.ones(len(P), dtype=bool)
    km = live_mask(K[:, L - C:L + H]) if prune else np.ones(len(K), dtype=bool)
    po, po_n = _subset(P[:, L - C:], pm, pn)
    kf, kf_n = _subset(K[:, L - C:L + H], km, kn)
    cases = [{'target_indices': list(range(N)), 'targets': Y[:, L - C:], 'past_only': po, 'known_future': kf, 'past_names': po_n, 'known_names': kf_n, 'provenance': 'primary|C=%d|past=%d|known=%d' % (C, len(po_n), len(kf_n))}]
    meta = [{'role': 'primary', 'offset': 0, 'C': C}]
    if n_aux > 0:
        for off, role in choose_offsets(L, H, stamps, n_aux=n_aux):
            e = L - off
            Cb = min(e, max_context)
            s = e - Cb
            if Cb < MIN_BACKTEST_CONTEXT or e + H > L:
                continue
            pmb = live_mask(P[:, s:e]) if prune else np.ones(len(P), dtype=bool)
            kmb = live_mask(K[:, s:e + H]) if prune else np.ones(len(K), dtype=bool)
            pob, pob_n = _subset(P[:, s:e], pmb, pn)
            kfb, kfb_n = _subset(K[:, s:e + H], kmb, kn)
            cases.append({'target_indices': list(range(N)), 'targets': Y[:, s:e], 'past_only': pob, 'known_future': kfb, 'past_names': pob_n, 'known_names': kfb_n, 'auxiliary': True, 'origin_offset': int(off), 'provenance': 'aux|%s|offset=%d|C=%d' % (role, off, Cb)})
            meta.append({'role': role, 'auxiliary': True, 'offset': int(off), 'C': Cb, 'truth_slice': [e, e + H]})
    state['case_meta'] = meta
    return cases

import numpy as np

def _seriation(C):
    n = C.shape[0]
    D = 1.0 - C
    np.fill_diagonal(D, 0.0)
    members = {i: [i] for i in range(n)}
    active = list(range(n))
    Dm = D.copy()
    while len(active) > 1:
        best = None
        for ai in range(len(active)):
            for aj in range(ai + 1, len(active)):
                d = Dm[active[ai], active[aj]]
                if best is None or d < best[0]:
                    best = (d, ai, aj)
        _, ai, aj = best
        i, j = (active[ai], active[aj])
        ni, nj = (len(members[i]), len(members[j]))
        for k in active:
            if k in (i, j):
                continue
            nd = (ni * Dm[i, k] + nj * Dm[j, k]) / (ni + nj)
            Dm[i, k] = Dm[k, i] = nd
        members[i] = members[i] + members[j]
        active.remove(j)
    return members[active[0]]

def grouped_indices(hist, max_vars=32, min_groups=None):
    n = hist.shape[0]
    if n <= max_vars:
        return [list(range(n))]
    win = min(hist.shape[1], 24 * 56)
    Z = hist[:, -win:]
    Z = np.diff(Z, axis=1)
    sd = Z.std(axis=1)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Zn = (Z - Z.mean(axis=1, keepdims=True)) / sd[:, None]
    C = Zn @ Zn.T / Zn.shape[1]
    C = np.clip(np.nan_to_num(C, nan=0.0), -1.0, 1.0)
    order = _seriation(C)
    g = int(np.ceil(n / max_vars))
    if min_groups:
        g = max(g, min_groups)
    sizes = [n // g + (1 if r < n % g else 0) for r in range(g)]
    out, p = ([], 0)
    for s in sizes:
        out.append(sorted(order[p:p + s]))
        p += s
    return out

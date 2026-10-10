import numpy as np

def regime_zscore(X, recent=336):
    cur = np.median(X[:, -recent:], axis=1)
    mu = X.mean(axis=1)
    sd = X.std(axis=1)
    return np.abs(cur - mu) / np.maximum(sd, 1e-09)

def regime_context(x, seas=48, z=None, z_thresh=1.5, tol_mult=3.0, min_ctx=1024, margin=2):
    L = x.shape[0]
    nd = L // seas
    if nd < 14:
        return L
    m = np.median(x[L - nd * seas:].reshape(nd, seas), axis=1)
    cur = np.median(m[-7:])
    mad = np.median(np.abs(np.diff(m[-14:]))) * 1.4826
    spread = np.subtract(*np.percentile(m, [75, 25]))
    tol = max(tol_mult * mad, 0.25 * spread, 1e-06)
    d = nd - 1
    while d >= 0 and abs(m[d] - cur) <= tol:
        d -= 1
    if d < 0:
        return L
    days = nd - 1 - d + margin
    return int(min(L, max(min_ctx, days * seas)))

def context_lengths(X, max_ctx, z_thresh=1.5, min_ctx=1024, tol_mult=3.0):
    D, L = X.shape
    full = min(L, max_ctx)
    z = regime_zscore(X)
    out = np.full(D, full, dtype=int)
    for i in range(D):
        if z[i] > z_thresh:
            out[i] = min(full, regime_context(X[i], min_ctx=min_ctx, tol_mult=tol_mult))
    return (out, z)

def bucket(lengths, full, grid=(1024, 1536, 2304, 3456, 5184)):
    out = np.empty_like(lengths)
    for i, v in enumerate(lengths):
        if v >= full:
            out[i] = full
            continue
        cand = [g for g in grid if g >= v]
        out[i] = min(full, cand[0] if cand else full)
    return out

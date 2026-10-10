import numpy as np

def stable_length(x, block=168, min_blocks=4, k_typ=6.0, k_range=0.15):
    L = x.size
    nb = L // block
    if nb < min_blocks:
        return L
    med = np.empty(nb)
    for j in range(nb):
        seg = x[L - (nb - j) * block:L - (nb - j - 1) * block]
        med[j] = np.median(seg)
    d = np.abs(np.diff(med))
    if d.size == 0:
        return L
    typ = np.median(d)
    rng = np.percentile(x, 95) - np.percentile(x, 5)
    thresh = max(k_typ * typ, k_range * rng)
    big = np.where(d > thresh)[0]
    if big.size == 0:
        return L
    return int((nb - (big[-1] + 1)) * block)

def context_buckets(H, L, levels=(672, 2016), pad=1.25, min_ctx=336):
    n = H.shape[0]
    want = np.empty(n)
    for i in range(n):
        want[i] = max(min_ctx, pad * stable_length(H[i]))
    chosen = np.empty(n, dtype=np.int64)
    for i in range(n):
        c = L
        for lv in levels:
            if want[i] <= lv:
                c = min(lv, L)
                break
        chosen[i] = c
    return chosen

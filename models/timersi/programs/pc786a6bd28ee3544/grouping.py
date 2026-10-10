import numpy as np
from robust import rank_corr

def univariate(n):
    return [[i] for i in range(n)]

def single_block(n):
    return [list(range(n))]

def correlation_blocks(H, max_size=26):
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    n = H.shape[0]
    if n <= 2:
        return single_block(n)
    C = rank_corr(H)
    D = 1.0 - np.abs(C)
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), 'average')
    for ncl in range(1, n + 1):
        lab = fcluster(Z, ncl, 'maxclust')
        sizes = np.bincount(lab)[1:]
        if sizes.max() <= max_size:
            break
    groups = [list(np.where(lab == c)[0]) for c in np.unique(lab)]
    return [g for g in groups if g]

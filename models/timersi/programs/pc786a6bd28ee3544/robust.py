import numpy as np

def nan_to_finite(x):
    x = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        x = np.where(np.isfinite(x), x, np.nan)
        for row in np.atleast_2d(x):
            idx = np.where(~np.isnan(row))[0]
            if idx.size == 0:
                row[:] = 0.0
                continue
            row[:idx[0]] = row[idx[0]]
            last = row[idx[0]]
            for j in range(idx[0], row.size):
                if np.isnan(row[j]):
                    row[j] = last
                else:
                    last = row[j]
    return x

def robust_center_scale(H, floor=1e-09):
    med = np.median(H, axis=1)
    mad = np.median(np.abs(H - med[:, None]), axis=1) * 1.4826
    iqr = (np.percentile(H, 75, axis=1) - np.percentile(H, 25, axis=1)) / 1.349
    sc = np.maximum(mad, iqr)
    bad = ~np.isfinite(sc) | (sc <= floor)
    if np.any(bad):
        alt = H.std(axis=1)
        sc = np.where(bad, np.maximum(alt, floor), sc)
    return (med, np.maximum(sc, floor))

def seasonal_scale(H, season=24):
    if H.shape[1] <= season:
        return np.maximum(np.abs(np.diff(H, axis=1)).mean(1), 1e-12)
    return np.maximum(np.abs(H[:, season:] - H[:, :-season]).mean(1), 1e-12)

def rank_corr(H, max_points=3000):
    X = H
    if X.shape[1] > max_points:
        X = X[:, -max_points:]
    R = np.apply_along_axis(lambda r: np.argsort(np.argsort(r)).astype(np.float64), 1, X)
    R -= R.mean(axis=1, keepdims=True)
    sd = R.std(axis=1)
    sd[sd <= 0] = 1.0
    R /= sd[:, None]
    C = R @ R.T / R.shape[1]
    np.fill_diagonal(C, 1.0)
    return np.clip(C, -1.0, 1.0)

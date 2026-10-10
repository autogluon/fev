import numpy as np
FLOOR = 1.0

def plan(target_history, observed=None):
    Y = np.asarray(target_history, dtype=float)
    use = []
    for d in range(Y.shape[0]):
        row = Y[d]
        if observed is not None:
            row = row[np.asarray(observed)[d].astype(bool)]
        row = row[np.isfinite(row)]
        use.append(bool(row.size > 0 and np.nanmin(row) > FLOOR))
    return np.asarray(use, dtype=bool)

def forward(target_history, use):
    Y = np.asarray(target_history, dtype=float).copy()
    for d in np.flatnonzero(use):
        Y[d] = np.log(np.maximum(Y[d], FLOOR))
    return Y

def inverse_point(x, use_d):
    return np.exp(np.asarray(x, dtype=float)) if use_d else np.asarray(x, dtype=float)

def inverse_quantiles(q, use_d):
    q = np.asarray(q, dtype=float)
    return np.exp(q) if use_d else q

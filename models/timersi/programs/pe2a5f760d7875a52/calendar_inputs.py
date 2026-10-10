import numpy as np
from calendar_ops import N_FEATURES, week_features
NAMES = ('cal_pay_cos', 'cal_pay_sin', 'cal_movable', 'cal_fixed')

def known_matrix(timestamps):
    X = week_features([str(t) for t in timestamps])
    out = np.stack([X[:, 0], X[:, 1], X[:, 4], X[:, 5]])
    return np.ascontiguousarray(out, dtype=float)

def augment(known_features, known_names, timestamps):
    cal = known_matrix(timestamps)
    names = list(known_names) + list(NAMES)
    if known_features is None or len(known_features) == 0:
        return (cal, names)
    base = np.asarray(known_features, float)
    if base.shape[1] != cal.shape[1]:
        return (base, list(known_names))
    return (np.vstack([base, cal]), names)

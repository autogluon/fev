import numpy as np
from calendar_ops import week_features
MOVABLE, FIXED = (4, 5)
SEEN_MASS = 1.0

def event_mass(timestamps, col):
    if len(timestamps) == 0:
        return 0.0
    return float(week_features([str(t) for t in timestamps])[:, col].sum())

def unseen_weight(ts_past, ts_future):
    Xf = week_features([str(t) for t in ts_future])
    w = np.zeros(len(ts_future))
    for col in (MOVABLE, FIXED):
        if event_mass(ts_past, col) < SEEN_MASS:
            w = np.maximum(w, Xf[:, col])
    return np.clip(w, 0.0, 1.0)

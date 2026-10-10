import numpy as np

def time_index(n_total, n_context):
    t = np.arange(n_total, dtype=float)
    c = t[:n_context]
    mu = c.mean()
    sd = c.std()
    if not np.isfinite(sd) or sd <= 0:
        sd = 1.0
    return (t - mu) / sd

def build(timestamps, n_context, horizon):
    n_total = n_context + horizon
    rows, names = ([], [])
    rows.append(time_index(n_total, n_context))
    names.append('calendar_time_index')
    return (np.asarray(rows, float), names)

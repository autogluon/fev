from __future__ import annotations
import numpy as np
POSITIVE = ('solar', 'entsoe', 'ercot', 'proenfo', 'rohlik', 'rossmann', 'favorita', 'm5_', 'hospital', 'restaurant', 'hierarchical_sales', 'australian_tourism', 'world_tourism', 'uk_covid')

def constraints(task, arrays):
    lengths = arrays['y_past_lengths']
    past = arrays['y_past']
    n, targets = (len(lengths), len(task.target_columns))
    positive = np.zeros((n, targets), bool)
    cumulative = positive.copy()
    last = np.zeros((n, targets))
    ends = np.cumsum(lengths)
    for item, end in enumerate(ends):
        history = past[end - lengths[item]:end]
        for target, name in enumerate(task.target_columns):
            observed = history[:, target]
            observed = observed[np.isfinite(observed)]
            if not len(observed):
                continue
            last[item, target] = observed[-1]
            positive[item, target] = task.task_name.startswith(POSITIVE) and np.min(observed) >= 0
            recent = observed[-max(64, 2 * task.horizon):]
            cumulative[item, target] = 'cumulative' in task.task_name and name.startswith('cumulative_') and (len(recent) >= 2) and np.all(np.diff(recent) >= 0)
    return (positive, cumulative, last)

def closed_store(window, task, ids):
    mask = np.zeros((len(ids), task.horizon, len(task.target_columns)), bool)
    if not task.dataset_config.startswith('rossmann') or 'Open' not in task.known_dynamic_columns:
        return mask
    past, future = window.get_input_data()
    before = {str(row[task.id_column]): row for row in past}
    after = {str(row[task.id_column]): row for row in future}
    for item, identifier in enumerate(ids):
        row = before[str(identifier)]
        closed = np.asarray(row['Open']) == 0
        if np.sum(closed) < 4:
            continue
        for target, name in enumerate(task.target_columns):
            values = np.asarray(row[name], float)[closed]
            values = values[np.isfinite(values)]
            if len(values) >= 4 and np.all(np.abs(values) <= 1e-08):
                mask[item, :, target] = np.asarray(after[str(identifier)]['Open']) == 0
    return mask

def project(point, quantiles, positive, cumulative, last, closed):

    def low(p, q):
        return (np.where(positive[:, None, :], np.maximum(p, 0), p), np.where(positive[:, None, :, None], np.maximum(q, 0), q))

    def mono(p, q):
        p = np.where(cumulative[:, None, :], np.maximum.accumulate(np.maximum(p, last[:, None, :]), axis=1), p)
        q = np.where(cumulative[:, None, :, None], np.maximum.accumulate(np.maximum(q, last[:, None, :, None]), axis=1), q)
        return (p, q)

    def shut(p, q):
        return (np.where(closed, 0, p), np.where(closed[..., None], 0, q))
    return {'causal78': (point, quantiles), 'positive': low(point, quantiles), 'known-closure': shut(point, quantiles), 'cumulative': mono(point, quantiles), 'all-constraints': shut(*mono(*low(point, quantiles)))}

def forecast(window, task, ids, arrays, pair):
    positive, cumulative, last = constraints(task, arrays)
    closed = closed_store(window, task, ids)
    return project(*pair, positive, cumulative, last, closed)

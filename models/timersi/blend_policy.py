from __future__ import annotations
import numpy as np

def apply(bank, groups, decision):
    names = {f'g{index:02d}::group' for index in range(int(groups.max()) + 1)}
    if set(decision.get('curves', {})) - names:
        raise ValueError('Decision names a group absent from this window')
    point, quantiles = (value.copy() for value in bank['baseline'])
    applied = {}
    for group in range(int(groups.max()) + 1):
        key = f'g{group:02d}::group'
        entry = decision.get('curves', {}).get(key, decision.get('default', {'weights': {'baseline': 1}}))
        weights = entry['weights']
        if not weights or len(weights) > 3 or set(weights) - set(bank):
            raise ValueError('Expected one to three available candidates')
        weights = {name: float(value) for name, value in weights.items()}
        if not all((np.isfinite(value) and value >= 0 for value in weights.values())) or sum(weights.values()) <= 0:
            raise ValueError('Expected finite nonnegative weights')
        mass = sum(weights.values())
        weights = {name: value / mass for name, value in weights.items()}
        item, target = np.where(groups == group)
        point[item, :, target] = sum((value * bank[name][0][item, :, target] for name, value in weights.items()))
        quantiles[item, :, target, :] = sum((value * bank[name][1][item, :, target, :] for name, value in weights.items()))
        applied[key] = weights
    assert np.isfinite(point).all() and np.isfinite(quantiles).all()
    assert np.all(np.diff(quantiles, axis=-1) >= -1e-08)
    return (point, quantiles, applied)

def decision_for_window(policy, task_name, window, start):
    task = next((entry for entry in policy['tasks'] if entry['task_name'] == task_name))
    for entry in task['query_decisions']:
        if entry['window'] == window:
            if np.datetime64(entry['start']) != np.datetime64(start):
                raise ValueError('FEV origin differs from frozen fixed policy')
            return entry['decision']
    return None

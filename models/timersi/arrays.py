from __future__ import annotations
import numpy as np

def curves(prediction, task):
    point = np.stack([np.asarray(prediction[name]['predictions'], float) for name in task.target_columns], axis=2)
    quantiles = np.stack([np.stack([np.asarray(prediction[name][str(level)], float) for level in task.quantile_levels], axis=-1) for name in task.target_columns], axis=2)
    return (point, quantiles)

def prediction_dataset(point, quantiles, task):
    import datasets
    point, quantiles = (np.asarray(point), np.asarray(quantiles))
    n, horizon, targets = point.shape
    if targets != len(task.target_columns) or horizon != task.horizon:
        raise ValueError('Point forecast does not match the FEV task')
    if quantiles.shape != (n, horizon, targets, len(task.quantile_levels)):
        raise ValueError('Quantile forecast does not match the FEV task')
    return datasets.DatasetDict({name: datasets.Dataset.from_dict({'predictions': point[:, :, target].tolist(), **{str(level): quantiles[:, :, target, index].tolist() for index, level in enumerate(task.quantile_levels)}}) for target, name in enumerate(task.target_columns)})

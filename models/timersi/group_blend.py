from __future__ import annotations
import json
from pathlib import Path
import warnings
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from constraints import constraints, closed_store, project
from blend_policy import apply, decision_for_window
HERE = Path(__file__).resolve().parent
SPECIALIST_TASKS = {'entsoe_15T', 'solar_with_weather_1H', 'epf_pjm', 'uk_covid_nation_1D/cumulative'}

def origins(num_windows):
    return sorted(set(np.linspace(0, num_windows - 1, min(4, num_windows), dtype=int).tolist()))

def groups_for(view, max_groups=12):
    n, horizon, targets = view.point.shape
    fields = np.concatenate([view.x.mean(axis=1), np.std(view.quantiles[..., 4], axis=1)[..., None] / view.unit[..., None]], axis=-1)
    x = np.clip(np.nan_to_num(fields.reshape(n * targets, -1)), -20, 20)
    if len(x) <= max_groups:
        labels = np.arange(len(x))
    else:
        center = np.median(x, axis=0)
        spread = np.maximum(np.median(abs(x - center), axis=0), 0.1)
        z = np.clip((x - center) / spread, -5, 5)
        model = MiniBatchKMeans(n_clusters=max_groups, n_init=3, batch_size=4096, max_iter=50, random_state=90)
        labels = model.fit_predict(z)
        remapped = np.empty_like(labels)
        for group, label in enumerate(sorted(set(labels))):
            remapped[labels == label] = group
        labels = remapped
    return labels.reshape(n, targets)

def bias_from_history(ids, unit, history):
    n, targets = unit.shape
    values = []
    for row in history[-3:]:
        lookup = {str(item): index for index, item in enumerate(row['ids'])}
        mapped = np.array([lookup.get(str(item), -1) for item in ids])
        source = row['residual']
        aligned = np.full((n, source.shape[1], targets), np.nan)
        valid = mapped >= 0
        aligned[valid] = source[mapped[valid]]
        values.append(aligned)
    if values:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            bias = np.nanmedian(np.concatenate(values, axis=1), axis=1)
        bias = 0.5 * np.clip(np.nan_to_num(bias) / unit, -5, 5) * unit
    else:
        bias = np.zeros_like(unit)
    return bias[:, None, :]

class GroupBlend:

    def __init__(self, task_name, policy=None):
        self.task_name = task_name
        self.policy = policy or json.loads((HERE / 'group_policy.json').read_text())
        self.history = []

    def predict(self, window, task, index, start, arrays, ids, original, raw, correction_bank, projected, specialized=None):
        from fev.metrics import _abs_seasonal_error_per_item
        from corrections import calendar
        from correction_core import make_view
        if self.task_name in SPECIALIST_TASKS and specialized is None:
            raise ValueError('Four fixed tasks require their live numeric-projected baseline')
        baseline = specialized if specialized is not None else projected
        bank = {'original': original, 'causal82': projected, 'baseline': baseline, 'raw': raw, 'shape': correction_bank['shape-blend'], 'ridge': correction_bank['conditional-ridge'], 'gbm': correction_bank['conditional-lgbm'], 'spread': correction_bank['calibrated-spread']}
        scale = _abs_seasonal_error_per_item(y_past=arrays['y_past'], y_past_lengths=arrays['y_past_lengths'], seasonality=task.seasonality)
        view = make_view(*baseline, arrays['y_past'], arrays['y_past_lengths'], scale, task.seasonality, calendar(window, task))
        mature = [row for row in self.history if np.datetime64(row['end']) < np.datetime64(start)]
        delta = bias_from_history(ids, view.unit, mature)
        bank['bias'] = (baseline[0] + delta, baseline[1] + delta[..., None])
        groups = groups_for(view)
        decision = decision_for_window(self.policy, self.task_name, index, start)
        if decision is None:
            return (baseline, {'ids': ids, 'baseline': baseline, 'bank': bank, 'groups': groups, 'bias_delta': delta, 'decision': None})
        revised = apply(bank, groups, decision)[:2]
        positive, cumulative, last = constraints(task, arrays)
        closed = closed_store(window, task, ids)
        final = project(*revised, positive, cumulative, last, closed)['all-constraints']
        return (final, {'ids': ids, 'baseline': baseline, 'bank': bank, 'groups': groups, 'bias_delta': delta, 'decision': decision})

    def observe(self, state, truth, end):
        self.history.append({'end': str(end), 'ids': np.asarray(state['ids'], str), 'residual': np.asarray(truth) - state['baseline'][1][..., 4]})

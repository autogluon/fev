from __future__ import annotations
import json
from pathlib import Path
import warnings
import numpy as np
from constraints import closed_store, constraints, project
from corrections import calendar
from correction_core import make_view
HERE = Path(__file__).resolve().parent
TASKS = {'entsoe_15T', 'solar_with_weather_1H', 'epf_pjm', 'uk_covid_nation_1D/cumulative'}

def padded(values, size, targets):
    output = np.full((len(values), size, targets), np.nan)
    for index, row in enumerate(values):
        row = np.asarray(row, float).reshape(-1, targets)[-size:]
        output[index, -len(row):] = row
    return output

def current_case(window, task, arrays, raw):
    from fev.metrics import _abs_seasonal_error_per_item
    point, quantiles = raw
    n, horizon, targets = point.shape
    lengths = arrays['y_past_lengths']
    ends = np.cumsum(lengths)
    histories = [arrays['y_past'][end - length:end] for end, length in zip(ends, lengths)]
    context = min(15360, max((len(row) for row in histories)))
    past_data, future_data = window.get_input_data()
    known, past_covariates = ([], [])
    known_names, past_names = ([], [])
    for name in task.known_dynamic_columns:
        try:
            known.append(np.asarray(future_data[name], float))
            known_names.append(name)
        except (TypeError, ValueError):
            pass
    for name in dict.fromkeys([*task.known_dynamic_columns, *task.past_dynamic_columns]):
        try:
            values = [np.asarray(row)[:, None] for row in past_data[name]]
            past_covariates.append(padded(values, context, 1)[..., 0])
            past_names.append(name)
        except (TypeError, ValueError):
            pass
    scale = _abs_seasonal_error_per_item(y_past=arrays['y_past'], y_past_lengths=lengths, seasonality=task.seasonality)
    return {'raw_point': point, 'raw_quantiles': quantiles, 'past': padded(histories, context, targets), 'past_lengths': lengths, 'scale': scale, 'calendar': calendar(window, task), 'known_future': np.stack(known, axis=-1) if known else np.empty((n, horizon, 0)), 'past_covariates': np.stack(past_covariates, axis=-1) if past_covariates else np.empty((n, context, 0)), 'known_names': known_names, 'past_names': past_names}

def dense_features(case, seasonality):
    point, quantiles = (case['raw_point'], case['raw_quantiles'])
    n, horizon, targets = point.shape
    context = case['past'].shape[1]
    lengths = np.minimum(case['past_lengths'], context)
    past = np.concatenate([case['past'][i, -int(length):] for i, length in enumerate(lengths)])
    view = make_view(point, quantiles, past, lengths, case['scale'], seasonality, case['calendar'])
    extra = [np.broadcast_to(value[:, None, :], point.shape) for value in (view.nonnegative.astype(float), view.monotone.astype(float))]
    extra.append(np.log1p(np.abs(quantiles[..., 4]) / view.unit[:, None, :]))
    extra.append(np.full(point.shape, np.log1p(seasonality)))
    aggregate = np.zeros(point.shape)
    mass = np.zeros((n, targets))
    for index, name in enumerate(case['known_names']):
        if name not in case['past_names']:
            continue
        x = case['past_covariates'][:, -min(context, 1024):, case['past_names'].index(name)]
        y = case['past'][:, -min(context, 1024):]
        xm, xs = (np.nanmean(x, axis=1), np.nanstd(x, axis=1))
        ym, ys = (np.nanmean(y, axis=1), np.nanstd(y, axis=1))
        corr = np.nanmean((x - xm[:, None])[..., None] * (y - ym[:, None, :]), axis=1) / np.maximum(xs[:, None] * ys, 1e-09)
        corr = np.nan_to_num(corr)
        z = (case['known_future'][..., index] - xm[:, None]) / np.maximum(xs[:, None], 1e-09)
        aggregate += corr[:, None, :] * np.nan_to_num(z)[..., None]
        mass += abs(corr)
    extra.append(aggregate / np.maximum(mass[:, None, :], 1))
    compact = np.clip(np.nan_to_num(np.concatenate([view.x, np.stack(extra, axis=-1)], axis=-1)), -20, 20)
    more = []
    span = max(1, min(context, int(seasonality)))
    for index, name in enumerate(case['known_names']):
        if name not in case['past_names']:
            continue
        previous = case['past_covariates'][..., case['past_names'].index(name)]
        mean = np.nanmean(previous, axis=1)
        scale = np.maximum(np.nanstd(previous, axis=1), 1e-09)
        future = case['known_future'][..., index]
        seasonal = previous[:, context - span + np.arange(horizon) % span]
        for value in ((future - mean[:, None]) / scale[:, None], (future - seasonal) / scale[:, None]):
            more.append(np.broadcast_to(value[..., None], (n, horizon, targets)))
    for index, name in enumerate(case['past_names']):
        if name in case['known_names']:
            continue
        previous = case['past_covariates'][..., index]
        value = (previous[:, -1] - np.nanmean(previous, axis=1)) / np.maximum(np.nanstd(previous, axis=1), 1e-09)
        more.append(np.broadcast_to(value[:, None, None], (n, horizon, targets)))
    features = np.concatenate([compact, np.stack(more, axis=-1)], axis=-1) if more else compact
    return (np.clip(np.nan_to_num(features), -20, 20), view.unit)

def apply_numeric(bank, ids, targets, decision):
    point, quantiles = (value.copy() for value in bank['baseline'])
    valid = {f'{item}::{target}' for item in ids for target in targets}
    if set(decision.get('curves', {})) - valid:
        raise ValueError('Unknown fixed curve key')
    for item, item_id in enumerate(ids):
        for target, name in enumerate(targets):
            entry = decision.get('curves', {}).get(f'{item_id}::{name}', decision.get('default', {'weights': {'baseline': 1}}))
            weights = {key: float(value) for key, value in entry['weights'].items()}
            if not weights or set(weights) - set(bank):
                raise ValueError('Unknown fixed candidate')
            total = sum(weights.values())
            point[item, :, target] = sum((weight / total * bank[key][0][item, :, target] for key, weight in weights.items()))
            quantiles[item, :, target, :] = sum((weight / total * bank[key][1][item, :, target, :] for key, weight in weights.items()))
    return (point, quantiles)

class Specialist:

    def __init__(self, task_name):
        if task_name not in TASKS:
            raise ValueError(task_name)
        self.task_name = task_name
        policy = json.loads((HERE / 'specialist_policy.json').read_text())
        self.decisions = {row['window']: row['decision'] for task in policy['tasks'] if task['task_name'] == task_name for row in task['decisions']}
        asset = json.loads((HERE / 'specialist_models' / (task_name.replace('/', '__') + '.json')).read_text())
        self.fits = {row['origin']: row['models'] for row in asset['fits']}
        self.models = None
        self.history = []

    def predict(self, window, task, index, start, arrays, ids, raw, baseline82):
        import lightgbm as lgb
        case = current_case(window, task, arrays, raw)
        features, unit = dense_features(case, task.seasonality)
        if index % 4 == 0:
            self.models = [lgb.Booster(model_str=value) for value in self.fits[index]]
        x = features.reshape(-1, features.shape[-1])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            delta = np.stack([model.predict(x, num_threads=2) for model in self.models], axis=-1)
        delta = delta.reshape(raw[1].shape) * unit[:, None, :, None]
        q = np.sort(raw[1] + delta, axis=-1)
        dense = (raw[0] + delta[..., 4], q)
        base_point, base_q = baseline82
        lengths = np.minimum(case['past_lengths'], case['past'].shape[1])
        past = np.concatenate([case['past'][i, -int(length):] for i, length in enumerate(lengths)])
        view = make_view(*baseline82, past, lengths, case['scale'], task.seasonality, case['calendar'])
        seasonal = 0.25 * (view.expert - base_q[..., 4])
        trend = 0.25 * view.x[..., 2] * view.unit[:, None, :]
        bias = np.zeros((len(ids), len(task.target_columns)))
        mature = [row for row in self.history if np.datetime64(row['end']) < np.datetime64(start)][-3:]
        for item, item_id in enumerate(ids):
            values = [row['truth'][row['ids'].index(str(item_id))] - row['baseline'][1][row['ids'].index(str(item_id)), ..., 4] for row in mature if str(item_id) in row['ids']]
            if values:
                bias[item] = 0.5 * np.clip(np.nanmedian(np.concatenate(values), axis=0) / view.unit[item], -5, 5) * view.unit[item]
        bias = np.nan_to_num(bias)[:, None, :]
        bank = {'baseline': baseline82, 'raw': raw, 'seasonal': (base_point + seasonal, base_q + seasonal[..., None]), 'trend': (base_point + trend, base_q + trend[..., None]), 'bias': (base_point + bias, base_q + bias[..., None]), 'dense': dense}
        pair = apply_numeric(bank, ids, task.target_columns, self.decisions[index]) if index in self.decisions else baseline82
        positive, cumulative, last = constraints(task, arrays)
        closed = closed_store(window, task, ids)
        final = project(*pair, positive, cumulative, last, closed)['all-constraints']
        state = {'ids': list(map(str, ids)), 'baseline': baseline82}
        return (final, state, {'features': features, 'dense': dense, 'numeric': pair})

    def observe(self, state, truth, end):
        self.history.append({'ids': state['ids'], 'baseline': state['baseline'], 'truth': np.asarray(truth), 'end': str(end)})

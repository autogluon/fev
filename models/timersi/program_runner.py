from __future__ import annotations
import collections
import importlib
import importlib.machinery
import importlib.util
from pathlib import Path
import sys
import time
import types
import numpy as np

def load_pipeline(folder):
    folder = Path(folder).resolve()
    for name, module in list(sys.modules.items()):
        filename = getattr(module, '__file__', None)
        local_name = name.split('.')[0]
        if filename and (Path(filename).resolve().is_relative_to(folder) or (folder / (local_name + '.py')).is_file() or (folder / local_name / '__init__.py').is_file()):
            del sys.modules[name]
    sys.path[:] = [str(folder), *[path for path in sys.path if not Path(path).resolve().is_relative_to(folder)]]
    package_name = f'_timersi_final_{time.time_ns()}'
    init = folder / '__init__.py'
    if init.exists():
        spec = importlib.util.spec_from_file_location(package_name, init, submodule_search_locations=[str(folder)])
        package = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = package
        spec.loader.exec_module(package)
    else:
        package = types.ModuleType(package_name)
        package.__file__ = str(init)
        package.__package__ = package_name
        package.__path__ = [str(folder)]
        package.__spec__ = importlib.machinery.ModuleSpec(package_name, loader=None, is_package=True)
        package.__spec__.submodule_search_locations = package.__path__
        sys.modules[package_name] = package
    return importlib.import_module(f'{package_name}.pipeline')

def encode(values):
    values = np.asarray(values)
    if values.dtype.kind in 'OUS':
        values = [float(abs(hash(value)) % 100) if isinstance(value, str) else 0.0 for value in values]
    return np.asarray(values, dtype=np.float32)

def item_view(past, future, task, item_index):
    from_timesfm = task._timersi_impute
    target = np.stack([np.asarray(past[name], dtype=np.float32) for name in task.target_columns])
    context_length = target.shape[1]
    known, known_names, historical, past_names = ([], [], [], [])
    for name in task.known_dynamic_columns:
        before, after = (from_timesfm(encode(past[name])), from_timesfm(encode(future[name])))
        if np.isnan(before).mean() <= 0.25 and np.isnan(after).mean() <= 0.25:
            known.append(np.concatenate([before, after]))
            known_names.append(name)
    for name in task.past_dynamic_columns:
        before = from_timesfm(encode(past[name]))
        if np.isnan(before).mean() <= 0.25:
            historical.append(before)
            past_names.append(name)
    return {'item_index': item_index, 'item_id': str(past[task.id_column]), 'target_history': np.stack([from_timesfm(series.copy()) for series in target]), 'target_observed': np.isfinite(target), 'target_ids': list(task.target_columns), 'target_names': list(task.target_columns), 'cutoff_index': context_length, 'horizon': task.horizon, 'timestamps': [str(ts) for ts in list(past[task.timestamp_column]) + list(future[task.timestamp_column])], 'past_features': np.stack(historical) if historical else np.empty((0, context_length), np.float32), 'past_names': past_names, 'known_features': np.stack(known) if known else np.empty((0, context_length + task.horizon), np.float32), 'known_names': known_names, 'static': {name: past[name] for name in task.static_columns}}

def task_card(task):
    return {'task_name': task.task_name, 'frequency': task.freq, 'horizon': task.horizon, 'seasonality': task.seasonality, 'targets': task.target_columns, 'known_dynamic_columns': task.known_dynamic_columns, 'past_dynamic_columns': task.past_dynamic_columns, 'static_columns': task.static_columns, 'semantic_source': 'Official FEV task name and actual dataset columns/static fields', 'semantic_source_url': f'https://huggingface.co/datasets/{task.dataset_path}', 'task_adaptation': 'Frozen task recipe', 'limits': {'max_context': 15360}, 'future_rule': 'Only FEV known dynamic columns, static fields and constructed calendar/causal estimates.'}

class ProgramRunner:

    def __init__(self, model, imputation):
        self.model = model
        self.forecaster = model._get_forecaster()
        self.imputation = imputation
        self.native_calls = 0

    def predict(self, window, task, code):
        module = load_pipeline(code)
        past, future = window.get_input_data()
        card = task_card(task)
        task._timersi_impute = self.imputation
        count, targets, horizon = (len(past), len(task.target_columns), task.horizon)
        points = np.empty((count, targets, horizon))
        quantiles = np.empty((count, targets, horizon, 9))
        for start in range(0, count, 64):
            items, buckets = ([], collections.defaultdict(list))
            for item in range(start, min(start + 64, count)):
                view = item_view(past[item], future[item], task, item)
                prepared, state = module.preprocess(view, card)
                variables, state = module.engineer(prepared, card, state)
                cases, state = module.select_context(variables, card, state)
                seen = []
                for case in cases:
                    values = np.asarray(case['targets'], np.float32)
                    indices = list(case['target_indices'])
                    auxiliary = bool(case.get('auxiliary', False))
                    alternative = bool(case.get('alternative', False))
                    offset = int(case.get('origin_offset', 0))
                    if auxiliary:
                        partial = bool(case.get('partial_auxiliary', False))
                        minimum = 1 if partial else horizon
                        if offset < minimum or offset >= view['cutoff_index'] or (partial and offset >= horizon):
                            raise ValueError('Auxiliary origin outside observed past')
                        if any((index not in range(targets) for index in indices)):
                            raise ValueError('Unknown auxiliary target')
                    elif alternative:
                        if any((index not in range(targets) for index in indices)):
                            raise ValueError('Unknown alternative target')
                    else:
                        seen += indices
                    context = values.shape[1]
                    available = view['cutoff_index'] - offset if auxiliary else view['cutoff_index']
                    if values.shape[0] != len(indices) or not 1 <= context <= min(available, 15360):
                        raise ValueError('Invalid context/target dimensions')
                    for name, length in (('past_only', context), ('known_future', context + horizon)):
                        covariate = case.get(name)
                        if covariate is not None:
                            covariate = np.asarray(covariate, np.float32)
                            if covariate.ndim != 2 or covariate.shape[1] != length or (not np.isfinite(covariate).all()):
                                raise ValueError(f'Invalid {name} covariate')
                            case[name] = covariate
                    if not np.isfinite(values).all():
                        raise ValueError('Non-finite target context')
                    case['targets'] = values
                if sorted(seen) != list(range(targets)):
                    raise ValueError('Every target must appear exactly once')
                outputs = [None] * len(cases)
                items.append((item, view, state, cases, outputs))
                for index, case in enumerate(cases):
                    shape = (len(case['target_indices']), 0 if case.get('past_only') is None else len(case['past_only']), 0 if case.get('known_future') is None else len(case['known_future']))
                    recent = bool(case.get('partial_auxiliary') or case.get('recent_partial_group'))
                    buckets[shape, recent].append((items[-1], index, case))
            for (shape, _recent), rows in buckets.items():
                maximum_context = max((row[2]['targets'].shape[1] for row in rows))
                size = self.model.get_optimal_batch_size(maximum_context, sum(shape))
                for begin in range(0, len(rows), size):
                    chunk = rows[begin:begin + size]
                    raw = list(self.forecaster.predict_batch(contexts=[row[2]['targets'] for row in chunk], horizon=horizon, past_only_covariates=[row[2].get('past_only') for row in chunk], past_future_covariates=[row[2].get('known_future') for row in chunk], return_quantiles=True, use_symmetric_averaging=True, make_positive=True, sort_quantiles=True))
                    self.native_calls += 1
                    for row, result in zip(chunk, raw):
                        point = np.asarray(result.forecast).reshape(shape[0], horizon)
                        quantile = np.asarray(result.quantiles).reshape(shape[0], horizon, 9)
                        row[0][4][row[1]] = {'point': point, 'quantiles': quantile}
            for item, _view, state, cases, outputs in items:
                result = module.postprocess(outputs, cases, state)
                point, quantile = (np.asarray(result['point']), np.asarray(result['quantiles']))
                if point.shape != (targets, horizon) or quantile.shape != (targets, horizon, 9):
                    raise ValueError('Invalid final point/quantile dimensions')
                points[item], quantiles[item] = (point, quantile)
        return (np.transpose(points, (0, 2, 1)), np.transpose(quantiles, (0, 2, 1, 3)))

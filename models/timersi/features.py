from __future__ import annotations
import ast
import hashlib
import numpy as np
import pandas as pd

def load_builder(program):
    source = program['source']
    if hashlib.sha256(source.encode()).hexdigest() != program['sha256']:
        raise ValueError('Frozen feature program hash changed')
    tree = ast.parse(source)
    if any((isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree))):
        raise ValueError('Frozen feature program unexpectedly imports a module')
    namespace = {'np': np, 'pd': pd}
    exec(compile(tree, '<evoltime-frozen-feature>', 'exec'), namespace)
    builder = namespace.get('features')
    if not callable(builder):
        raise ValueError('Frozen feature program lacks features()')
    return builder

def build_channels(builder, past, future, task):
    context = len(past[task.timestamp_column])
    dates = np.concatenate([np.asarray(row[task.timestamp_column], dtype='datetime64[ns]') for row in (past, future)])
    covariates = {name: np.concatenate([np.asarray(row[name], float) for row in (past, future)]) for name in task.known_dynamic_columns}
    values = builder(dates, covariates, context)
    if not isinstance(values, dict) or not 1 <= len(values) <= 12:
        raise ValueError('Feature program must return 1–12 channels')
    channels = {'agent_' + str(name): np.asarray(value, dtype=np.float32) for name, value in values.items()}
    for name, value in channels.items():
        if value.shape != (len(dates),) or not np.isfinite(value).all():
            raise ValueError(f'{name}: expected finite, aligned one-dimensional channel')
    return ({name: value[:context].tolist() for name, value in channels.items()}, {name: value[context:].tolist() for name, value in channels.items()})

class FeatureWindow:

    def __init__(self, window, task, program):
        self.window = window
        self.task = task
        self.builder = load_builder(program)
        self.names = None
        self._input = None

    def __getattr__(self, name):
        return getattr(self.window, name)

    def get_input_data(self):
        if self._input is None:
            past, future = self.window.get_input_data()
            pairs = [build_channels(self.builder, a, b, self.task) for a, b in zip(past, future)]
            self.names = list(pairs[0][0])
            if not all((list(pair[0]) == self.names and list(pair[1]) == self.names for pair in pairs)):
                raise ValueError('Feature channel names differ between entities')
            self._input = tuple((data.map(lambda row, index: {**row, **pairs[index][side]}, with_indices=True, load_from_cache_file=False) for side, data in enumerate((past, future))))
        return self._input

class SingleWindow:

    def __init__(self, task, window, extra_known):
        self.task = task
        self.window = window
        self.known_dynamic_columns = [*task.known_dynamic_columns, *extra_known]
        self.past_dynamic_columns = list(task.past_dynamic_columns)

    def __getattr__(self, name):
        return getattr(self.task, name)

    def iter_windows(self, *args, **kwargs):
        yield self.window

def forecast_new_pool(model, task, window, programs, incumbent, prior, raw, curves):
    pool = {'incumbent': incumbent, 'prior': prior, 'raw': raw}
    for program in programs:
        native = forecast_native(model, task, window, program, curves)
        pool[program['name']] = tuple((0.75 * base + 0.25 * addition for base, addition in zip(prior, native)))
    return pool

def forecast_native(model, task, window, program, curves):
    wrapped = FeatureWindow(window, task, program)
    wrapped.get_input_data()
    prediction = model._fit_predict(SingleWindow(task, wrapped, wrapped.names))[0]
    return curves(prediction, task)

def forecast_selected_blend(model, task, window, programs, base, curves):
    if not programs:
        return base
    natives = [forecast_native(model, task, window, program, curves) for program in programs]
    return tuple((0.75 * base[index] + 0.25 * np.mean([pair[index] for pair in natives], axis=0) for index in (0, 1)))

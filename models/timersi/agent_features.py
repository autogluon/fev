"""Execute the frozen task-specific numerical covariate programs."""
import ast
import builtins
import hashlib

import numpy as np
import pandas as pd

from features import SingleWindow


def numeric(values):
    try:
        return np.asarray(values, float)
    except (TypeError, ValueError):
        return np.asarray([
            int.from_bytes(hashlib.blake2b(str(value).encode(), digest_size=4).digest(), 'big') / 2**32
            for value in values
        ], float)


def load_builder(source, expected_hash):
    if hashlib.sha256(source.encode()).hexdigest() != expected_hash:
        raise ValueError('Frozen feature source changed')
    tree = ast.parse(source)
    if any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree)):
        raise ValueError('Feature source must use supplied np and pd')
    namespace = {'np': np, 'pd': pd, '__builtins__': builtins.__dict__}
    exec(compile(tree, '<frozen-features>', 'exec'), namespace)
    return namespace['features']


def build(builder, past, future, task):
    context = len(past[task.timestamp_column])
    dates = np.concatenate([
        np.asarray(row[task.timestamp_column], dtype='datetime64[ns]')
        for row in (past, future)
    ])
    covariates = {
        name: numeric(list(past[name]) + list(future[name]))
        for name in task.known_dynamic_columns
    }
    values = builder(dates, covariates, context)
    if not isinstance(values, dict) or not 1 <= len(values) <= 12:
        raise ValueError('Expected 1–12 numerical feature arrays')
    arrays = {'agent_' + str(name): np.asarray(value, dtype=np.float32)
              for name, value in values.items()}
    for name, value in arrays.items():
        if value.shape != (len(dates),) or not np.isfinite(value).all():
            raise ValueError(f'{name}: expected finite aligned feature')
    return ({name: value[:context].tolist() for name, value in arrays.items()},
            {name: value[context:].tolist() for name, value in arrays.items()})


class AgentFeatureWindow:
    def __init__(self, window, task, program):
        self.window = window
        self.task = task
        self.code_hash = program['sha256']
        self.builder = load_builder(program['source'], self.code_hash)
        self.names = None
        self._input = None

    def __getattr__(self, name):
        return getattr(self.window, name)

    def get_input_data(self):
        if self._input is None:
            past, future = self.window.get_input_data()
            columns = [self.task.timestamp_column, *self.task.known_dynamic_columns]
            compact = [data.select_columns(columns).with_format('numpy')[:]
                       for data in (past, future)]
            pairs = [build(self.builder,
                           {name: compact[0][name][index] for name in columns},
                           {name: compact[1][name][index] for name in columns},
                           self.task)
                     for index in range(len(past))]
            self.names = list(pairs[0][0])
            if not all(list(pair[0]) == self.names and list(pair[1]) == self.names for pair in pairs):
                raise ValueError('Feature names differ between entities')
            inputs = []
            for side, data in enumerate((past, future)):
                for name in self.names:
                    fingerprint = hashlib.sha256(
                        f'{data._fingerprint}:{self.code_hash}:{side}:{name}'.encode()
                    ).hexdigest()
                    data = data.add_column(name, [pair[side][name] for pair in pairs],
                                           new_fingerprint=fingerprint)
                inputs.append(data)
            self._input = tuple(inputs)
        return self._input


def forecast(model, task, window, program, curves):
    wrapped = AgentFeatureWindow(window, task, program)
    wrapped.get_input_data()
    return curves(model._fit_predict(SingleWindow(task, wrapped, wrapped.names))[0], task)

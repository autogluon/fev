from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from arrays import curves
from group_blend import bias_from_history, groups_for
from constraints import closed_store, constraints, project
from blend_policy import apply, decision_for_window
from corrections import calendar
from correction_core import make_view
HERE = Path(__file__).resolve().parent
TASKS = {'uk_covid_nation_1D/cumulative', 'uk_covid_nation_1W/cumulative', 'uk_covid_utla_1W/cumulative'}

class SingleWindow:

    def __init__(self, task, window, extra_past=()):
        self.task, self.window = (task, window)
        self.past_dynamic_columns = [*task.past_dynamic_columns, *extra_past]

    def __getattr__(self, name):
        return getattr(self.task, name)

    def iter_windows(self, *args, **kwargs):
        yield self.window

class FlowWindow:

    def __init__(self, window, task, add_covariate_flows=False):
        self.window, self.task = (window, task)
        self.extra_past = ['flow__' + name for name in task.past_dynamic_columns if name.startswith('cumulative_')] if add_covariate_flows else []
        self._input = None

    def __getattr__(self, name):
        return getattr(self.window, name)

    def get_input_data(self):
        if self._input is None:
            past, future = self.window.get_input_data()
            target_columns = list(self.task.target_columns)
            dynamic = list(dict.fromkeys([self.task.timestamp_column, *target_columns, *self.task.past_dynamic_columns, *self.task.known_dynamic_columns]))

            def transform(row):
                result = dict(row)
                for name in dynamic:
                    result[name] = row[name][1:]
                for name in target_columns:
                    result[name] = np.diff(np.asarray(row[name], float)).tolist()
                for name in self.extra_past:
                    result[name] = np.diff(np.asarray(row[name[len('flow__'):]], float)).tolist()
                return result
            self._input = (past.map(transform, load_from_cache_file=False, desc='Observed stock-to-flow'), future)
        return self._input

def integrate(point, quantiles, last, uncertainty):
    p = last[:, None, :] + np.cumsum(point, axis=1)
    median = last[:, None, :] + np.cumsum(quantiles[..., 4], axis=1)
    deviations = quantiles - quantiles[..., 4:5]
    if uncertainty == 'sum':
        interval = np.cumsum(deviations, axis=1)
    elif uncertainty == 'rss':
        interval = np.sign(np.arange(1, 10) / 10 - 0.5) * np.sqrt(np.cumsum(deviations ** 2, axis=1))
    else:
        raise ValueError(uncertainty)
    return (p, median[..., None] + interval)

def native_bank(model, window, task):
    past, _future = window.get_input_data()
    last = np.array([[np.asarray(row[name], float)[np.isfinite(np.asarray(row[name], float))][-1] for name in task.target_columns] for row in past])
    output = {}
    flow_pair = None
    for kind in ('flow', 'flow-cov'):
        if kind == 'flow-cov' and (not any((name.startswith('cumulative_') for name in task.past_dynamic_columns))):
            pair = flow_pair
        else:
            flow = FlowWindow(window, task, add_covariate_flows=kind == 'flow-cov')
            pair = curves(model._fit_predict(SingleWindow(task, flow, flow.extra_past))[0], task)
        if kind == 'flow':
            flow_pair = pair
        for uncertainty in ('rss', 'sum'):
            output[kind + '-' + uncertainty] = integrate(*pair, last, uncertainty)
    return output

class FlowBlend:

    def __init__(self, task_name, model):
        if task_name not in TASKS:
            raise ValueError(task_name)
        self.task_name = task_name
        self.model = model
        self.policy = json.loads((HERE / 'flow_policy.json').read_text())
        self.history = []

    def predict(self, window, task, index, start, arrays, ids, raw, base, lower_bank):
        from fev.metrics import _abs_seasonal_error_per_item
        if decision_for_window(self.policy, self.task_name, index, start) is None:
            return (base, {'ids': list(map(str, ids)), 'baseline': base})
        scale = _abs_seasonal_error_per_item(y_past=arrays['y_past'], y_past_lengths=arrays['y_past_lengths'], seasonality=task.seasonality)
        bank = dict(lower_bank)
        bank['baseline'] = base
        bank.update(native_bank(self.model, window, task))
        view = make_view(*base, arrays['y_past'], arrays['y_past_lengths'], scale, task.seasonality, calendar(window, task))
        mature = [row for row in self.history if np.datetime64(row['end']) < np.datetime64(start)]
        delta = bias_from_history(ids, view.unit, mature)
        bank['bias'] = (base[0] + delta, base[1] + delta[..., None])
        groups = groups_for(view)
        decision = decision_for_window(self.policy, self.task_name, index, start)
        revised = apply(bank, groups, decision)[:2] if decision else base
        positive, cumulative, last = constraints(task, arrays)
        closed = closed_store(window, task, ids)
        final = project(*revised, positive, cumulative, last, closed)['all-constraints']
        return (final, {'ids': list(map(str, ids)), 'baseline': base, 'groups': groups, 'bank': bank, 'decision': decision})

    def observe(self, state, truth, end):
        self.history.append({'ids': state['ids'], 'end': str(end), 'residual': np.asarray(truth) - state['baseline'][1][..., 4]})

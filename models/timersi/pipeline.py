import gzip
import json
from pathlib import Path
import numpy as np
from arrays import curves, prediction_dataset
from constraints import forecast as project
from corrections import Corrections
from features import SingleWindow, forecast_native, forecast_selected_blend
from flow import FlowBlend, TASKS as FLOW_TASKS
from group_blend import GroupBlend
from native import LastValueImputation
from program_runner import ProgramRunner
from specialist import Specialist, TASKS as SPECIALIST_TASKS
from transforms import asinh_native, calendar_native
HERE = Path(__file__).resolve().parent

def recipes():
    with gzip.open(HERE / 'recipes.json.gz', 'rt') as stream:
        return json.load(stream)

def blend(base, additions, weight=0.25):
    if not additions:
        return base
    return tuple(((1 - weight) * base[k] + weight * np.mean([pair[k] for pair in additions], axis=0) for k in (0, 1)))

class FrozenPipeline:

    def __init__(self, task, model):
        self.task, self.model = (task, model)
        self.recipe = next((row for row in recipes()['tasks'] if row['task_name'] == task.task_name))
        self.features = json.loads((HERE / 'features.json').read_text())
        self.runner = ProgramRunner(model, LastValueImputation())
        self.corrections = Corrections(task.task_name)
        self.specialist = Specialist(task.task_name) if task.task_name in SPECIALIST_TASKS else None
        self.groups = GroupBlend(task.task_name)
        self.flows = FlowBlend(task.task_name, model) if task.task_name in FLOW_TASKS else None

    def predict_window(self, window, index):
        task, rule = (self.task, self.recipe['windows'][index])
        past, future = window.get_input_data()
        start = str(min((np.datetime64(row[task.timestamp_column][0]) for row in future)))
        end = str(max((np.datetime64(row[task.timestamp_column][-1]) for row in future)))
        if np.datetime64(start) != np.datetime64(rule['start']) or np.datetime64(end) != np.datetime64(rule['end']):
            raise ValueError('The supplied forecast window differs from the frozen recipe')
        raw = curves(self.model._fit_predict(SingleWindow(task, window, []))[0], task)
        alpha = rule['program_weight']
        if alpha:
            program = self.runner.predict(window, task, HERE / 'programs' / self.recipe['program'])
            original = tuple(((1 - alpha) * a + alpha * b for a, b in zip(raw, program)))
        else:
            original = raw
        if rule['semantic_program']:
            original = blend(original, [self.runner.predict(window, task, HERE / 'programs' / rule['semantic_program'])])
        bank, correction_state = self.corrections.predict(window, task, original, raw, start, rule)
        arrays, ids, _ = correction_state
        projected = project(window, task, ids, arrays, bank['online-top3'])['all-constraints']
        specialist_state = None
        specialized = None
        if self.specialist is not None:
            specialized, specialist_state, _ = self.specialist.predict(window, task, index, start, arrays, ids, raw, projected)
        base, group_state = self.groups.predict(window, task, index, start, arrays, ids, original, raw, bank, projected, specialized=specialized)
        flow_state = None
        if self.flows is not None:
            base, flow_state = self.flows.predict(window, task, index, start, arrays, ids, raw, base, group_state['bank'])
        if self.recipe['asinh']:
            base = blend(base, [asinh_native(self.model, window, task)])
        if self.recipe['calendar']:
            base = blend(base, [calendar_native(self.model, window, task)])
        native = {}

        def feature(key):
            if key not in native:
                native[key] = forecast_native(self.model, task, window, self.features[key], curves)
            return native[key]
        prior = blend(base, [feature(key) for key in self.recipe['prior_features']])
        incumbent = blend(prior, [feature(key) for key in self.recipe['portfolio_features']])
        action = rule['selection']
        pool = {'incumbent': incumbent, 'prior': prior, 'raw': raw}
        if 'chosen' in action:
            for candidate in set(action['chosen']):
                if candidate.startswith('native:'):
                    pool[candidate] = blend(prior, [feature(candidate.split(':', 1)[1])])
            if list(map(str, ids)) != action['item_ids'] or len(task.target_columns) != action['targets']:
                raise ValueError('Input series or target order differs from the frozen recipe')
            point, quantiles = (value.copy() for value in incumbent)
            choices = np.asarray(action['chosen'], object).reshape(len(ids), len(task.target_columns))
            for candidate in set(action['chosen']):
                item, target = np.where(choices == candidate)
                point[item, :, target] = pool[candidate][0][item, :, target]
                quantiles[item, :, target] = pool[candidate][1][item, :, target]
            final = (point, quantiles)
        else:
            final = pool[action['default']]
        self.corrections.observe(correction_state, index, end)
        self.groups.observe(group_state, arrays['y_true'], end)
        if self.specialist is not None:
            self.specialist.observe(specialist_state, arrays['y_true'], end)
        if self.flows is not None:
            self.flows.observe(flow_state, arrays['y_true'], end)
        return prediction_dataset(*final, task)

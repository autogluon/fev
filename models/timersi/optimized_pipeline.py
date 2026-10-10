import numpy as np
import group_blend
from arrays import curves, prediction_dataset
from constraints import forecast as project
from features import SingleWindow, forecast_native
from pipeline import FrozenPipeline, HERE, blend
from transforms import asinh_native, calendar_native


class LazyGroups(group_blend.GroupBlend):
    def predict(self, window, task, index, start, *args, **kwargs):
        # Group labels are neither used for the no-decision result nor observe().
        # Called from one execution thread only; do not use this patch in threads.
        original = group_blend.groups_for
        if group_blend.decision_for_window(self.policy, self.task_name, index, start) is None:
            group_blend.groups_for = lambda view: None
        try:
            return super().predict(window, task, index, start, *args, **kwargs)
        finally:
            group_blend.groups_for = original


def selected_features(base, raw, action, prior_keys, portfolio_keys, feature, ids, targets):
    needed = set(action['chosen']) if 'chosen' in action else {action['default']}
    prior = blend(base, [feature(key) for key in prior_keys]) if needed - {'raw'} else base
    pool = {'raw': raw, 'prior': prior}
    if 'incumbent' in needed:
        pool['incumbent'] = blend(prior, [feature(key) for key in portfolio_keys])
    for candidate in sorted(needed):
        if candidate.startswith('native:'):
            pool[candidate] = blend(prior, [feature(candidate.split(':', 1)[1])])
    if 'chosen' not in action:
        return pool[action['default']]
    if list(map(str, ids)) != action['item_ids'] or targets != action['targets']:
        raise ValueError('Input series or target order differs from the frozen recipe')
    # Every item/target is assigned, so an unselected incumbent is not needed.
    point, quantiles = (value.copy() for value in raw)
    choices = np.asarray(action['chosen'], object).reshape(len(ids), targets)
    for candidate in sorted(needed):
        item, target = np.where(choices == candidate)
        point[item, :, target] = pool[candidate][0][item, :, target]
        quantiles[item, :, target] = pool[candidate][1][item, :, target]
    return point, quantiles


class LazyPipeline(FrozenPipeline):
    def __init__(self, task, model):
        super().__init__(task, model)
        self.groups = LazyGroups(task.task_name)

    def predict_window(self, window, index):
        task, rule = self.task, self.recipe['windows'][index]
        past, future = window.get_input_data()
        start = str(min(np.datetime64(row[task.timestamp_column][0]) for row in future))
        end = str(max(np.datetime64(row[task.timestamp_column][-1]) for row in future))
        if np.datetime64(start) != np.datetime64(rule['start']) or np.datetime64(end) != np.datetime64(rule['end']):
            raise ValueError('The supplied forecast window differs from the frozen recipe')
        raw = curves(self.model._fit_predict(SingleWindow(task, window, []))[0], task)
        alpha = rule['program_weight']
        if alpha:
            program = self.runner.predict(window, task, HERE / 'programs' / self.recipe['program'])
            original = tuple((1 - alpha) * a + alpha * b for a, b in zip(raw, program))
        else:
            original = raw
        if rule['semantic_program']:
            original = blend(original, [self.runner.predict(window, task, HERE / 'programs' / rule['semantic_program'])])
        bank, correction_state = self.corrections.predict(window, task, original, raw, start, rule)
        arrays, ids, _ = correction_state
        projected = project(window, task, ids, arrays, bank['online-top3'])['all-constraints']
        specialist_state, specialized = None, None
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

        final = selected_features(base, raw, rule['selection'], self.recipe['prior_features'],
                                  self.recipe['portfolio_features'], feature, ids, len(task.target_columns))
        # Preserve all feedback and chronological updates, including unselected baselines.
        self.corrections.observe(correction_state, index, end)
        self.groups.observe(group_state, arrays['y_true'], end)
        if self.specialist is not None:
            self.specialist.observe(specialist_state, arrays['y_true'], end)
        if self.flows is not None:
            self.flows.observe(flow_state, arrays['y_true'], end)
        return prediction_dataset(*final, task)

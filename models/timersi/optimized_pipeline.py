import json
import numpy as np
import group_blend
from agent_features import forecast as forecast_agent_feature
from arrays import curves, prediction_dataset
from constraints import forecast as project
from features import SingleWindow
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


class LazyPipeline(FrozenPipeline):
    def __init__(self, task, model):
        super().__init__(task, model)
        self.groups = LazyGroups(task.task_name)
        self.agent_program = json.loads((HERE / 'agent_features.json').read_text())['tasks'].get(task.task_name)

    def predict_window(self, window, index):
        task, rule = self.task, self.recipe['windows'][index]
        past, future = window.get_input_data()
        start = str(min(np.datetime64(row[task.timestamp_column][0]) for row in future))
        end = str(max(np.datetime64(row[task.timestamp_column][-1]) for row in future))
        if np.datetime64(start) != np.datetime64(rule['start']) or np.datetime64(end) != np.datetime64(rule['end']):
            raise ValueError('The supplied forecast window differs from the frozen recipe')
        raw = curves(self.model._fit_predict(SingleWindow(task, window, []))[0], task)
        if self.agent_program == 'raw':
            return prediction_dataset(*raw, task)
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
        if self.agent_program:
            final = blend(base, [forecast_agent_feature(self.model, task, window,
                                                        self.agent_program, curves)])
        else:
            final = base
        # Preserve all feedback and chronological updates, including unselected baselines.
        self.corrections.observe(correction_state, index, end)
        self.groups.observe(group_state, arrays['y_true'], end)
        if self.specialist is not None:
            self.specialist.observe(specialist_state, arrays['y_true'], end)
        if self.flows is not None:
            self.flows.observe(flow_state, arrays['y_true'], end)
        return prediction_dataset(*final, task)

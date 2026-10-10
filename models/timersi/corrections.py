import numpy as np
import pandas as pd
from fev.metrics import _abs_seasonal_error_per_item
from arrays import prediction_dataset
from correction_core import Reviser, make_view

def calendar(window, task):
    _, future = window.get_input_data()
    shape = (len(future), task.horizon)
    stamps = pd.DatetimeIndex(np.asarray(future[task.timestamp_column], dtype='datetime64[ns]').reshape(-1))
    channels = []
    for code, period in ((stamps.dayofweek, 7), (stamps.hour, 24), (stamps.month - 1, 12)):
        angle = np.asarray(code) * 2 * np.pi / period
        channels.extend([np.sin(angle).reshape(shape), np.cos(angle).reshape(shape)])
    return np.stack(channels, axis=-1)

class Corrections:

    def __init__(self, task_name):
        self.core = Reviser(task_name)

    def predict(self, window, task, pair, raw, start, recipe):
        arrays, ids = window._prepare_arrays(prediction_dataset(*pair, task), task.quantile_levels)
        scale = _abs_seasonal_error_per_item(y_past=arrays['y_past'], y_past_lengths=arrays['y_past_lengths'], seasonality=task.seasonality)
        view = make_view(*pair, arrays['y_past'], arrays['y_past_lengths'], scale, task.seasonality, calendar(window, task))
        bank, info = self.core.predict(view, start, recipe)
        bank['raw'] = raw
        return (bank, (arrays, ids, view))

    def observe(self, state, index, end):
        arrays, _, view = state
        self.core.observe(view, arrays['y_true'], index, end, {})

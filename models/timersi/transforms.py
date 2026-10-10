import numpy as np
import pandas as pd
from arrays import curves
from features import SingleWindow
CALENDAR_NAMES = ('calendar_hour_sin', 'calendar_hour_cos', 'calendar_week_sin', 'calendar_week_cos', 'calendar_year_sin', 'calendar_year_cos', 'calendar_weekend')

class CalendarWindow:

    def __init__(self, window, task):
        self.window, self.task = (window, task)
        self._input = None

    def __getattr__(self, name):
        return getattr(self.window, name)

    def get_input_data(self):
        if self._input is None:

            def augment(row):
                d = pd.DatetimeIndex(row[self.task.timestamp_column])
                hour = np.asarray(d.hour + d.minute / 60 + d.second / 3600, float)
                week = np.asarray(d.dayofweek, float) + hour / 24
                year = (np.asarray(d.dayofyear, float) - 1 + hour / 24) / np.where(d.is_leap_year, 366.0, 365.0)
                values = [v for phase in (hour / 24, week / 7, year) for v in (np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase))]
                values.append(np.asarray(d.dayofweek >= 5, float))
                return {**row, **{name: value.tolist() for name, value in zip(CALENDAR_NAMES, values)}}
            self._input = tuple((data.map(augment, load_from_cache_file=False) for data in self.window.get_input_data()))
        return self._input

class AsinhWindow:

    def __init__(self, window, task):
        self.window, self.task = (window, task)
        self.scales = []
        self._input = None

    def __getattr__(self, name):
        return getattr(self.window, name)

    def get_input_data(self):
        if self._input is None:
            past, future = self.window.get_input_data()

            def transform(row):
                result, scales = (dict(row), [])
                for name in self.task.target_columns:
                    values = np.asarray(row[name], float)
                    scale = max(1.0, float(np.nanmedian(abs(values))))
                    scales.append(scale)
                    result[name] = np.arcsinh(values / scale).tolist()
                return result
            self.scales = np.asarray([[max(1.0, float(np.nanmedian(abs(np.asarray(row[name], float))))) for name in self.task.target_columns] for row in past])
            self._input = (past.map(transform, load_from_cache_file=False), future)
        return self._input

def calendar_native(model, window, task):
    wrapped = CalendarWindow(window, task)
    return curves(model._fit_predict(SingleWindow(task, wrapped, CALENDAR_NAMES))[0], task)

def asinh_native(model, window, task):
    wrapped = AsinhWindow(window, task)
    point, quantiles = curves(model._fit_predict(SingleWindow(task, wrapped, []))[0], task)
    return (wrapped.scales[:, None, :] * np.sinh(point), wrapped.scales[:, None, :, None] * np.sinh(quantiles))

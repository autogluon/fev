"""Reuse the same FEV split inside a forecast window, never across cutoffs."""
import dataclasses
from fev.task import EvaluationWindow


class CachedWindow(EvaluationWindow):
    def _get_past_future_test_data(self):
        if not hasattr(self, '_timersi_split'):
            self._timersi_split = super()._get_past_future_test_data()
        return self._timersi_split


def cache_window(window):
    return CachedWindow(**{field.name: getattr(window, field.name)
                           for field in dataclasses.fields(EvaluationWindow)})

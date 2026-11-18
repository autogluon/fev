"""Keep the issued time grid when GluonTS cannot infer it from a short prefix."""

from contextlib import contextmanager
from unittest.mock import patch


@contextmanager
def toto_short_history(task, case):
    import pandas as pd
    from gluonts.dataset.pandas import PandasDataset

    _, past, future, *_ = case
    timestamps = task.timestamp_column
    history = past[0][timestamps]
    if len(history) >= 3:
        yield
        return
    issue_grid = pd.DatetimeIndex([*history, *future[0][timestamps]])
    frequency = pd.infer_freq(issue_grid) if len(issue_grid) >= 3 else task.freq
    if frequency is None:
        yield
        return
    constructor = PandasDataset.from_long_dataframe

    def with_frequency(*args, **kwargs):
        if kwargs.get("freq") is None:
            kwargs["freq"] = pd.tseries.frequencies.get_period_alias(frequency) or frequency
        return constructor(*args, **kwargs)

    with patch.object(PandasDataset, "from_long_dataframe", side_effect=with_frequency):
        yield

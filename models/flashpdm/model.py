"""Flashmind PDM through the public flashpdm client.

https://www.flashmind.com/models/pdm

Targets are joint unless known-future covariates require one target per input.
Numeric dynamic covariates retain their roles. String covariates use smoothed
per-series target means fitted on observed history only; static columns are unused.
"""

from itertools import tee
from typing import ClassVar

import datasets
import numpy as np
import pyarrow as pa
from flashpdm import Client
from flashpdm.batching import BatchLimits, ForecastPool

import fev

REVISIONS = {
    "pdm-femto": "46bb0bde1c836a6b1a82e6d0108f766adb47c8639cf9c695637c583a617fe06c.2026-10-27",
    "pdm-pico": "cf380714c0e6eda4a43a1e62a12cf228b3c0795fa21be7078ddaf6b866c501bd.2026-10-27",
    "pdm-nano": "48cc3fd4918228d313179822d616fca5c282c5bcc9e59d55bc33c31b2c98eb5e.2026-10-27",
}


def _numeric(values):
    return np.asarray(values, dtype=np.float64)


def _values(values):
    a = _numeric(values)
    if np.isinf(a).any():
        raise ValueError("Infinite observations are not supported")
    return np.where(np.isnan(a), None, a).tolist()


def _target_encode(target, past, future=None):
    """Smoothed category means (one pseudo-observation); unseen -> history mean."""

    def key(value):
        return None if value is None or value == "" or (isinstance(value, float) and np.isnan(value)) else str(value)

    y = _numeric(target)
    observed = y[np.isfinite(y)]
    mean = float(observed.mean()) if len(observed) else 0.0
    stats = {}
    for value, category in zip(y, past, strict=True):
        if np.isfinite(value):
            k = key(category)
            total, count = stats.get(k, (0.0, 0))
            stats[k] = (total + value, count + 1)

    def encode(values):
        result = []
        for value in values:
            total, count = stats.get(key(value), (0.0, 0))
            result.append((mean + total) / (1 + count))
        return np.asarray(result, dtype=np.float32)

    return encode(past), None if future is None else encode(future)


def _project(quantiles, history):
    """Infer nonnegative/count support from the observed history, never labels."""
    observed = _numeric(history)
    observed = observed[np.isfinite(observed)]
    if len(observed) and (observed >= 0).all():
        quantiles = np.maximum(quantiles, 0)
        if len(observed) >= 8 and (observed == np.rint(observed)).all():
            quantiles = np.rint(quantiles)
    return quantiles


class FlashPDMModel(fev.ForecastingModel):
    model_name = "flashpdm"
    trained_on_datasets: ClassVar[list[str]] = []

    def __init__(self, model="pdm-nano", revision=None, concurrency=32, batch_size=16):
        super().__init__()
        if model not in REVISIONS:
            raise ValueError(f"Choose one of {list(REVISIONS)}")
        if not 1 <= concurrency <= 256 or not 1 <= batch_size <= 32:
            raise ValueError("concurrency must be 1..256 and batch_size 1..32")
        self.model = model
        self.revision = revision or REVISIONS[model]
        self.concurrency = concurrency
        self.batch_size = batch_size

    def _items(self, task, windows, categorical, cap):
        for w, (past_data, future_data) in enumerate(windows):
            for i, (past, future) in enumerate(zip(past_data, future_data, strict=True)):
                groups = [[c] for c in task.target_columns] if task.known_dynamic_columns else [task.target_columns]
                for columns in groups:
                    if len(columns) > 1 and categorical:
                        raise ValueError(
                            "Joint targets with categorical covariates require an explicit encoding policy"
                        )
                    targets = {c: _numeric(past[c]) for c in columns}
                    p, f = {}, {}
                    for c in task.known_dynamic_columns + task.past_dynamic_columns:
                        ahead = future[c] if c in task.known_dynamic_columns else None
                        hist = past[c]
                        if c in categorical:
                            hist, ahead = _target_encode(targets[columns[0]], hist, ahead)
                        p[c] = _values(_numeric(hist)[-cap:])
                        if ahead is not None:
                            f[c] = _values(ahead)
                    payload = {"targets": {c: _values(y[-cap:]) for c, y in targets.items()}}
                    if p:
                        payload["past_covariates"] = p
                    if f:
                        payload["future_covariates"] = f
                    yield (w, i, targets, payload)

    def _fit_predict(self, task):
        levels = sorted({0.5, *task.quantile_levels})
        table = task.load_full_dataset(num_proc=1).data.table
        categorical = set()
        for c in task.dynamic_columns:
            dtype = table.schema.field(c).type.value_type
            if not (pa.types.is_integer(dtype) or pa.types.is_floating(dtype) or pa.types.is_boolean(dtype)):
                categorical.add(c)
        windows = [window.get_input_data() for window in task.iter_windows()]
        bands = [
            {c: np.full((len(past), task.horizon, len(levels)), np.nan) for c in task.target_columns}
            for past, _ in windows
        ]

        with Client(timeout=180, max_retries=2) as client, ForecastPool(
            client, model=self.model, revision=self.revision, concurrency=self.concurrency,
            limits=BatchLimits(windows=self.batch_size), max_attempts=3,
        ) as pool:
            items, metadata = tee(self._items(task, windows, categorical, pool.capabilities["max_context"]))
            with self._record_inference_time():
                forecasts = pool.forecast_inputs(
                    (item[3] for item in items), task.horizon, quantile_levels=levels,
                )
                for (w, i, targets, _), result in zip(metadata, forecasts, strict=True):
                    for c, history in targets.items():
                        q = np.asarray(result[c].quantile_values).T
                        if q.shape != (task.horizon, len(levels)) or not np.isfinite(q).all():
                            raise ValueError("Invalid forecast shape or non-finite prediction")
                        bands[w][c][i] = _project(q, history)

        predictions = []
        for window in bands:
            per_target = {}
            for c, q in window.items():
                if not np.isfinite(q).all():
                    raise ValueError("Missing predictions; refusing to export incomplete results")
                values = {"predictions": q[..., levels.index(0.5)]}
                values.update({str(level): q[..., levels.index(level)] for level in task.quantile_levels})
                per_target[c] = datasets.Dataset(
                    pa.table(
                        {
                            name: pa.FixedSizeListArray.from_arrays(pa.array(a.reshape(-1)), task.horizon)
                            for name, a in values.items()
                        }
                    )
                )
            predictions.append(datasets.DatasetDict(per_target))
        return predictions

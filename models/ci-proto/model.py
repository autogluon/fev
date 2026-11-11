"""CI-Proto on fev-bench.

CI-Proto is Continuous Intelligence's time-series foundation model. It runs through the
``continuousintelligence`` package, which loads the checkpoint and its auxiliary files
from the Hugging Face Hub and applies preprocessing and calibration.
"""

from typing import ClassVar
from uuid import uuid4

import continuousintelligence as ci
import datasets
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch

import fev

_PAD_TIME = np.iinfo(np.int64).min
_LEVEL_TOLERANCE = 1e-6
_MAX_CONTEXT = 16_384


def _is_text(table: pa.Table, column: str) -> bool:
    """Whether a sequence column holds text rather than numbers or true/false values."""
    value_type = table.schema.field(column).type.value_type
    return not (pa.types.is_floating(value_type) or pa.types.is_integer(value_type) or pa.types.is_boolean(value_type))


def _text_values(table: pa.Table, column: str) -> tuple[str, ...]:
    """Return the sorted unique values of a text sequence column, compared as text."""
    values = pc.list_flatten(table.column(column)).unique().to_pylist()
    return tuple(sorted(str(v) for v in values if v is not None))


def _rows(table: pa.Table, column: str, text_values: tuple[str, ...] | None = None) -> list[np.ndarray]:
    """Split a sequence column into one float64 array per series.

    A text column is coded by each value's position in ``text_values``, with missing text as NaN.
    """
    lists = table.column(column).combine_chunks()
    offsets = lists.offsets.to_numpy()
    flat = lists.flatten()
    if text_values is not None:
        flat = pc.index_in(flat.cast(pa.string()), value_set=pa.array(text_values, pa.string())).cast(pa.float64())
    values = flat.to_numpy(zero_copy_only=False).astype(np.float64)
    return np.split(values, offsets[1:-1] - offsets[0])


def _timestamps(table: pa.Table, column: str) -> list[np.ndarray]:
    """Split the timestamp column into one array of nanoseconds since the epoch per series."""
    lists = table.column(column).combine_chunks()
    offsets = lists.offsets.to_numpy()
    values = lists.flatten().cast(pa.timestamp("ns")).to_numpy().astype(np.int64)
    return np.split(values, offsets[1:-1] - offsets[0])


def _left_padded(rows: list[np.ndarray], fill: float) -> np.ndarray:
    """Stack rows of different lengths, padding each at the start with ``fill``."""
    length = max(len(row) for row in rows)
    stacked = np.full((len(rows), length), fill, dtype=rows[0].dtype)
    for i, row in enumerate(rows):
        stacked[i, length - len(row) :] = row
    return stacked


class CIProtoModel(fev.ForecastingModel):
    """CI-Proto, loaded from a Hugging Face repository at a pinned ``revision``."""

    model_name = "ci-proto"
    trained_on_datasets: ClassVar[list[str]] = []

    def __init__(
        self, model_path: str = "ContinuousIntelligence/ci-proto", revision: str | None = None, device: str = "cuda"
    ):
        super().__init__()
        self._model = ci.load_model(model_path, device=device, revision=revision)
        self._model.forecast(torch.sin(torch.arange(512, dtype=torch.float32) / 8)[None], horizon=32)

    def _fit_predict(self, task: fev.Task) -> list[datasets.DatasetDict]:
        full = task.load_full_dataset().with_format("arrow")[:]
        model_levels = self._model.quantiles.tolist()
        columns = []
        for level in task.quantile_levels:
            matches = [i for i, candidate in enumerate(model_levels) if abs(candidate - level) < _LEVEL_TOLERANCE]
            if not matches:
                raise ValueError(f"task {task.task_name!r} needs quantile level {level}, which the model lacks")
            columns.append(matches[0])
        text_values = {
            name: _text_values(full, name)
            for name in task.known_dynamic_columns + task.past_dynamic_columns
            if _is_text(full, name)
        }
        predictions = []
        for window in task.iter_windows():
            past_data, future_data = window.get_input_data()
            inputs = self._prepare_window(task, past_data, future_data, text_values)
            with self._record_inference_time():
                quantiles = self._forecast_window(task, inputs)
            predictions.append(self._predictions(task, quantiles[:, :, columns].double().numpy()))
        return predictions

    def _prepare_window(
        self,
        task: fev.Task,
        past_data: datasets.Dataset,
        future_data: datasets.Dataset,
        text_values: dict[str, tuple[str, ...]],
    ) -> list[dict[str, object]]:
        """Convert a window into the package's CPU inputs, preserving series and target order."""
        past = past_data.with_format("arrow")[:]
        future = future_data.with_format("arrow")[:]
        if not past.column(task.id_column).equals(future.column(task.id_column)):
            raise ValueError(f"task {task.task_name!r} gives its past and future data in different series orders")
        known, past_only = task.known_dynamic_columns, task.past_dynamic_columns
        histories = [
            np.stack(rows)[:, -_MAX_CONTEXT:] for rows in zip(*(_rows(past, name) for name in task.target_columns))
        ]
        past_covariates = [
            np.stack(rows)[:, -_MAX_CONTEXT:]
            for rows in zip(*(_rows(past, name, text_values.get(name)) for name in past_only))
        ]
        future_covariates = [
            np.stack([np.concatenate([h[-_MAX_CONTEXT:], f]) for h, f in zip(history, ahead)])
            for history, ahead in zip(
                zip(*(_rows(past, name, text_values.get(name)) for name in known)),
                zip(*(_rows(future, name, text_values.get(name)) for name in known)),
            )
        ]
        times = [
            np.concatenate([p[-_MAX_CONTEXT:], f])
            for p, f in zip(_timestamps(past, task.timestamp_column), _timestamps(future, task.timestamp_column))
        ]
        options = {
            "categorical_past": tuple(i for i, name in enumerate(past_only) if name in text_values),
            "categorical_future": tuple(i for i, name in enumerate(known) if name in text_values),
            "freq": task.freq,
        }

        if len(task.target_columns) == 1:
            history = _left_padded([h[0] for h in histories], np.nan)
            if past_only:
                padded = [_left_padded([c[k] for c in past_covariates], np.nan) for k in range(len(past_only))]
                options["past_covariates"] = torch.from_numpy(np.stack(padded, axis=1))
            if known:
                padded = [_left_padded([c[k] for c in future_covariates], np.nan) for k in range(len(known))]
                options["future_covariates"] = torch.from_numpy(np.stack(padded, axis=1))
            static = {name: past.column(name).to_pylist() for name in task.static_columns}
            return [
                {
                    "history": torch.from_numpy(history),
                    "timestamps": torch.from_numpy(_left_padded(times, _PAD_TIME)),
                    "series_ids": [str(i) for i in past.column(task.id_column).to_pylist()],
                    "static_features": static or None,
                    **options,
                }
            ]

        inputs = []
        for i, history in enumerate(histories):
            if past_only:
                options["past_covariates"] = torch.from_numpy(past_covariates[i])
            if known:
                options["future_covariates"] = torch.from_numpy(future_covariates[i])
            inputs.append({"history": torch.from_numpy(history), "timestamps": torch.from_numpy(times[i]), **options})
        return inputs

    def _forecast_window(self, task: fev.Task, inputs: list[dict[str, object]]) -> torch.Tensor:
        """Forecast every target of every series, as ``[series, targets, quantiles, horizon]``."""
        forecasts = [self._model.forecast(horizon=task.horizon, **item) for item in inputs]
        if len(task.target_columns) == 1:
            return forecasts[0][:, None]
        return torch.stack(forecasts)

    def _predictions(self, task: fev.Task, quantiles: np.ndarray) -> datasets.DatasetDict:
        """fev's predictions for one window, from quantiles at the task's levels."""
        levels = list(task.quantile_levels)
        if 0.5 not in levels:
            raise ValueError(f"task {task.task_name!r} has no 0.5 quantile level for the point forecast")
        names = ["predictions"] + [str(level) for level in levels]
        indices = [levels.index(0.5)] + list(range(len(levels)))
        info = datasets.DatasetInfo(
            features=datasets.Features({name: datasets.Sequence(datasets.Value("float64")) for name in names})
        )
        offsets = pa.array(np.arange(quantiles.shape[0] + 1, dtype=np.int32) * quantiles.shape[-1])
        predictions = {}
        for j, target in enumerate(task.target_columns):
            columns = {
                name: pa.ListArray.from_arrays(offsets, pa.array(quantiles[:, j, i].ravel(), type=pa.float64()))
                for name, i in zip(names, indices)
            }
            # Give transient predictions a unique cache ID without hashing the entire table.
            predictions[target] = datasets.Dataset(pa.table(columns), info=info, fingerprint=uuid4().hex)
        return datasets.DatasetDict(predictions)

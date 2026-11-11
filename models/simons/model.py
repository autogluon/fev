"""Simons (https://github.com/Abel-ai-lab/Simons) through fev's forecasting interface.

    python models/evaluate.py -m simons -n Simons-297M \
        -k '{"model": "abel-lab/Simons-297M", "batch_size": 256}' -b benchmarks/fev_bench/tasks.yaml

`model` selects a released model: abel-lab/Simons-25M, -60M, -178M or -297M. Targets are forecast
jointly in groups of at most 32, with up to 16 known-future covariates; categorical history and
horizon are coded separately. The median is the point forecast. On CUDA the trunk runs in bf16."""

from typing import ClassVar

import fev
import numpy as np

LEVELS = np.arange(1, 10) / 10  # the quantile levels Simons forecasts


class SimonsModel(fev.ForecastingModel):
    model_name = "simons"
    # The training data holds none of the fev-bench datasets (see the model cards).
    trained_on_datasets: ClassVar[list[str]] = []

    def __init__(self, model: str = "abel-lab/Simons-297M", device: str = "auto",
                 batch_size: int = 256):
        super().__init__()
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self._simons = None

    def _fit_predict(self, task: fev.Task) -> list[dict[str, list[dict]]]:
        """Forecast every window of the task; fev turns each {target: records} dict into a
        DatasetDict."""
        import torch

        from simons import Simons

        if self._simons is None:  # the weights are loaded once and kept across tasks
            device = self.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self._simons = Simons.from_pretrained(self.model, device=device)
        predictions_per_window = []
        for window in task.iter_windows():
            past_data, future_data = window.get_input_data()
            with self._record_inference_time():
                predictions_per_window.append(self._predict_window(task, past_data, future_data))
        return predictions_per_window

    def _predict_window(self, task: fev.Task, past_data, future_data) -> dict[str, list[dict]]:
        """Forecasts for one window: per target column, one record per item with the median and
        each quantile."""
        from simons import forecast_series

        horizon, targets = task.horizon, task.target_columns
        future_by_id = {}
        if future_data is not None:
            future_by_id = {row[task.id_column]: row for row in future_data}
        series, future_covariates, past_covariates = [], [], []
        for item in past_data:
            series.append(np.stack([np.asarray(item[t], dtype=np.float32) for t in targets]))
            future_row = future_by_id.get(item[task.id_column])
            known = {}
            for column in task.known_dynamic_columns:
                if future_row is not None and len(future_row[column]) == horizon:
                    known[column] = np.concatenate([np.asarray(item[column], dtype=object),
                                                    np.asarray(future_row[column], dtype=object)])
            future_covariates.append(known or None)
            past = {c: np.asarray(item[c], dtype=object) for c in task.past_dynamic_columns}
            past_covariates.append(past or None)
        outputs = forecast_series(
            self._simons,
            series,
            horizon,
            future_covariates=future_covariates,
            past_covariates=past_covariates,
            shared_categories=False,
            batch_size=self.batch_size,
        )
        quantiles = np.stack(outputs)  # [items, variates, horizon, 9]: items are passed as [V, T]
        predictions = {}
        for k, target in enumerate(targets):
            records = []
            for q in np.sort(quantiles[:, k], axis=-1):  # [horizon, 9]
                record = {"predictions": q[:, 4].tolist()}
                for level in task.quantile_levels:
                    record[str(level)] = [float(np.interp(level, LEVELS, step)) for step in q]
                records.append(record)
            predictions[target] = records
        return predictions

import datasets
import numpy as np

import fev


class TimeH2OModel(fev.ForecastingModel):
    """TimeH2O model from https://pypi.org/project/h2ofm/.

    Each item is passed as a (targets, history) series with its known dynamic covariates spanning history + horizon.
    timeh2o-v1 forecasts every target from its own history plus those covariates; past-only and static covariates
    are not used. The model emits quantiles only, so the point forecast is the median.
    """

    model_name = "timeh2o-v1"
    trained_on_datasets = []

    def __init__(
        self,
        device: str = "auto",
        batch_size: int = 256,
        context_length: int | None = None,
        clip_nonnegative: bool = True,
        snap_integer: bool = True,
        flip_equivariant: bool = True,
    ):
        super().__init__()
        self.device = device
        self.batch_size = batch_size
        self.context_length = context_length
        self.clip_nonnegative = clip_nonnegative
        self.snap_integer = snap_integer
        self.flip_equivariant = flip_equivariant
        self._model = None

    def _load_model(self):
        if self._model is None:
            from h2ofm import TimeH2O

            self._model = TimeH2O.from_pretrained(
                "h2oai/timeh2o-v1", device=self.device, batch_size=self.batch_size
            )
        return self._model

    def _fit_predict(self, task: fev.Task) -> list[datasets.DatasetDict]:
        model = self._load_model()
        predictions_per_window = []
        for window in task.iter_windows():
            past_data, future_data = window.get_input_data()
            past_data = past_data.with_format("numpy")
            future_data = future_data.with_format("numpy")
            target_columns = window.target_columns
            known_columns = window.known_dynamic_columns

            contexts = [
                np.stack([np.asarray(row[col], dtype=np.float32) for col in target_columns])
                for row in past_data.select_columns(target_columns)
            ]
            if known_columns:
                past_known = past_data.select_columns(known_columns)
                future_known = future_data.select_columns(known_columns)
                future_covariates = [
                    [np.concatenate([past_row[col], future_row[col]]) for col in known_columns]
                    for past_row, future_row in zip(past_known, future_known)
                ]
            else:
                future_covariates = None

            with self._record_inference_time():
                forecasts = model.forecast_many(
                    contexts,
                    horizon=window.horizon,
                    future_covariates=future_covariates,
                    quantile_levels=task.quantile_levels,
                    context_length=self.context_length,
                    clip_nonnegative=self.clip_nonnegative,
                    snap_integer=self.snap_integer,
                    flip_equivariant=self.flip_equivariant,
                )

            point = np.stack([f.point for f in forecasts])  # (items, targets, horizon)
            quantiles = np.stack([f.quantiles for f in forecasts])  # (items, targets, horizon, levels)
            predictions = {}
            for j, col in enumerate(target_columns):
                target_predictions = {"predictions": point[:, j]}
                for i, q in enumerate(task.quantile_levels):
                    target_predictions[str(q)] = quantiles[:, j, :, i]
                predictions[col] = datasets.Dataset.from_dict(target_predictions)
            predictions_per_window.append(datasets.DatasetDict(predictions))
        return predictions_per_window

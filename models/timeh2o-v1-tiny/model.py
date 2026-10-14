import datasets
import numpy as np

import fev


class TimeH2OModel(fev.ForecastingModel):
    """TimeH2O model from https://pypi.org/project/h2ofm/.

    timeh2o-v1 checkpoints forecast every target independently and ignore covariates, so tasks are
    converted to univariate series. The model emits quantiles only, so the point forecast is the median.
    """

    model_name = "timeh2o-v1-tiny"
    trained_on_datasets = []

    def __init__(
        self,
        device: str = "auto",
        batch_size: int = 256,
        context_length: int | None = None,
        clip_nonnegative: bool = True,
        flip_equivariant: bool = False,
    ):
        super().__init__()
        self.device = device
        self.batch_size = batch_size
        self.context_length = context_length
        self.clip_nonnegative = clip_nonnegative
        self.flip_equivariant = flip_equivariant
        self._model = None

    def _load_model(self):
        if self._model is None:
            from h2ofm import TimeH2O

            self._model = TimeH2O.from_pretrained(
                "h2oai/timeh2o-v1-tiny", device=self.device, batch_size=self.batch_size
            )
        return self._model

    def _fit_predict(self, task: fev.Task) -> list[datasets.DatasetDict]:
        model = self._load_model()
        predictions_per_window = []
        for window in task.iter_windows():
            past_data, _ = fev.convert_input_data(window, adapter="datasets", as_univariate=True)
            past_data = past_data.with_format("numpy").cast_column(
                "target", datasets.Sequence(datasets.Value("float32"))
            )
            contexts = list(past_data["target"])

            with self._record_inference_time():
                forecasts = model.forecast_many(
                    contexts,
                    horizon=window.horizon,
                    quantile_levels=task.quantile_levels,
                    context_length=self.context_length,
                    clip_nonnegative=self.clip_nonnegative,
                    flip_equivariant=self.flip_equivariant,
                )

            quantiles = np.stack([f.quantiles for f in forecasts])  # (series, horizon, levels)
            predictions = {"predictions": np.stack([f.point for f in forecasts])}
            for i, q in enumerate(task.quantile_levels):
                predictions[str(q)] = quantiles[:, :, i]
            predictions_per_window.append(
                fev.combine_univariate_predictions_to_multivariate(
                    datasets.Dataset.from_dict(predictions), target_columns=window.target_columns
                )
            )
        return predictions_per_window

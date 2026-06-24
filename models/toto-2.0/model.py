import logging
import warnings

import datasets
import numpy as np

import fev


class Toto2Model(fev.ForecastingModel):
    """Toto 2.0 model from https://github.com/DataDog/toto (https://pypi.org/project/toto-2)."""

    model_name = "toto-2.0"
    trained_on_datasets = []

    def __init__(
        self,
        model_path: str = "Datadog/Toto-2.0-22m",
        batch_size: int = 512,
        context_length: int = 4096,
        decode_block_size: int | None = None,
        as_univariate: bool = False,
        device: str = "auto",
    ):
        super().__init__()
        self.model_path = model_path
        self.batch_size = batch_size
        self.context_length = context_length
        self.decode_block_size = decode_block_size
        self.as_univariate = as_univariate
        self.device = device

    def _fit_predict(self, task: fev.Task) -> list[datasets.DatasetDict]:
        import torch
        from toto2 import Toto2GluonTSModel, Toto2GluonTSModelConfig
        from toto2 import Toto2Model as PretrainedToto2

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        target_columns = ["target"] if self.as_univariate else task.target_columns

        model = PretrainedToto2.from_pretrained(fev.utils.maybe_cache_from_s3(self.model_path))
        config = Toto2GluonTSModelConfig(
            prediction_length=task.horizon,
            context_length=self.context_length,
            target_dim=len(target_columns),
            decode_block_size=self.decode_block_size,
            quantiles=task.quantile_levels,
        )
        gts_model = Toto2GluonTSModel(model.to(self.device).eval(), config)
        predictor = gts_model.create_predictor(batch_size=self.batch_size, device=self.device)

        logging.getLogger("gluonts").setLevel(100)
        # The 0.5 quantile is used as the point forecast (Toto 2.0 is quantile-based and has no mean prediction).
        forecast_keys = {"predictions": 0.5, **{str(q): q for q in task.quantile_levels}}

        predictions_per_window = []
        for window in task.iter_windows():
            _, prediction_dataset = fev.convert_input_data(window, adapter="gluonts", as_univariate=self.as_univariate)
            with self._record_inference_time():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    forecasts = list(predictor.predict(prediction_dataset))
            predictions_per_window.append(self._format_predictions(forecasts, task, forecast_keys))
        return predictions_per_window

    def _format_predictions(
        self, forecasts: list, task: fev.Task, forecast_keys: dict[str, float]
    ) -> datasets.DatasetDict:
        """Format GluonTS forecasts into a `DatasetDict` keyed by target column, as expected by fev."""
        if self.as_univariate:
            # One univariate forecast per (item, variate), interleaved by variate; `f.quantile(q)` is (horizon,).
            flat = datasets.Dataset.from_dict(
                {key: np.stack([f.quantile(q) for f in forecasts]) for key, q in forecast_keys.items()}
            )
            return fev.utils.combine_univariate_predictions_to_multivariate(flat, target_columns=task.target_columns)
        else:
            # One forecast per item, reshaped to (num_items, horizon, n_variates). The model squeezes the
            # variate axis for single-target tasks, so `f.quantile(q)` is (horizon,) there and (horizon, n_var) else.
            quantiles = {
                key: np.stack([f.quantile(q).reshape(task.horizon, -1) for f in forecasts])
                for key, q in forecast_keys.items()
            }
            return datasets.DatasetDict(
                {
                    col: datasets.Dataset.from_dict({key: arr[..., i] for key, arr in quantiles.items()})
                    for i, col in enumerate(task.target_columns)
                }
            )

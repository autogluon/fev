"""
TiRex-2 was evaluated on a NVIDIA H100 GPU with CUDA version 12.6 installed. 
"""
import contextlib

import datasets
import fev

from tirex2 import ForecastModel, load_model


class TiRex2Model(fev.ForecastingModel):
    """TiRex model from https://github.com/NX-AI/tirex-2."""

    model_name = "tirex-2"
    trained_on_datasets = []

    def __init__(
        self,
        model_path: str = "NX-AI/TiRex-2-fevbench",
        batch_size: int = 512,
        device: str = "cuda",
        as_univariate: bool = False,
    ):
        super().__init__()
        self.model_path = model_path
        self.batch_size = batch_size
        self.device = device
        self.as_univariate = as_univariate
        self._model: ForecastModel | None = None

    def _load_model(self) -> ForecastModel:
        if self._model is None:
            load_device = "cuda" if self.device.startswith("cuda") else self.device
            self._model = load_model(fev.utils.maybe_cache_from_s3(self.model_path), device=load_device)
            self._model.model.to(self.device)
        return self._model

    def _fit_predict(self, task: fev.Task) -> list[datasets.DatasetDict]:
        # Necessary to avoid infinite recursion during custom kernel compilation inside SageMaker
        contextlib.redirect_stdout = lambda _: contextlib.nullcontext()

        if self.inference_time is None:
            self.inference_time = 0

        model = self._load_model()
        predictions_per_window = []
        for window in task.iter_windows():
            forecast, inference_time_s = model.forecast_fev(
                window=window,
                prediction_length=window.horizon,
                output_type="fev",
                quantile_levels=task.quantile_levels,
                batch_size=self.batch_size,
                data_kwargs={"as_univariate": self.as_univariate},
                return_inference_time=True,
            )
            predictions_per_window.append(forecast)
            self.inference_time += inference_time_s

        return predictions_per_window

import datasets
import torch
from chronos import BaseChronosPipeline

import fev
from fev.model import ForecastingModel


class ChronosModel(ForecastingModel):
    """Chronos-Bolt model from https://github.com/amazon-science/chronos-forecasting."""

    model_name = "chronos"

    def __init__(
        self,
        model_path: str = "amazon/chronos-bolt-base",
        device_map: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        batch_size: int = 32,
    ):
        super().__init__()
        self.model_path = model_path
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.batch_size = batch_size

    def _fit_predict(self, task: fev.Task) -> list[datasets.DatasetDict]:
        pipeline = BaseChronosPipeline.from_pretrained(
            self.model_path, device_map=self.device_map, torch_dtype=self.torch_dtype
        )

        predictions_per_window, self.inference_time = pipeline.predict_fev(task, batch_size=self.batch_size)

        return predictions_per_window

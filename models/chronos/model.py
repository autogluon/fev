import datasets

import fev


class ChronosModel(fev.ForecastingModel):
    """Chronos-Bolt model from https://github.com/amazon-science/chronos-forecasting."""

    model_name = "chronos"

    def __init__(
        self,
        model_path: str = "amazon/chronos-bolt-base",
        device: str = "cuda",
        batch_size: int = 256,
    ):
        super().__init__()
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size

    def _fit_predict(self, task: fev.Task) -> list[datasets.DatasetDict]:
        import torch
        from chronos import BaseChronosPipeline

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        pipeline = BaseChronosPipeline.from_pretrained(
            self.model_path, device_map=self.device, torch_dtype=torch.float32
        )

        predictions_per_window, self.inference_time = pipeline.predict_fev(task, batch_size=self.batch_size)

        return predictions_per_window

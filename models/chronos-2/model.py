import datasets

import fev


class Chronos2Model(fev.ForecastingModel):
    """Chronos-2 model from https://github.com/amazon-science/chronos-forecasting."""

    model_name = "chronos-2"

    def __init__(
        self,
        model_path: str = "amazon/chronos-2",
        device: str = "cuda",
        batch_size: int = 100,
        cross_learning: bool = True,
    ):
        super().__init__()
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size
        self.cross_learning = cross_learning

    def _fit_predict(self, task: fev.Task) -> list[datasets.DatasetDict]:
        import torch
        from chronos import BaseChronosPipeline

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        pipeline = BaseChronosPipeline.from_pretrained(
            self.model_path, device_map=self.device, torch_dtype=torch.float32
        )

        predictions_per_window, self.inference_time = pipeline.predict_fev(
            task, batch_size=self.batch_size, cross_learning=self.cross_learning
        )

        return predictions_per_window

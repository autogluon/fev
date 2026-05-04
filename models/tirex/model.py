import contextlib
import os

import datasets

import fev


class TiRexModel(fev.ForecastingModel):
    """TiRex model from https://github.com/NX-AI/tirex."""

    model_name = "tirex"

    def __init__(
        self,
        model_path: str = "NX-AI/TiRex",
        batch_size: int = 512,
        device: str = "cuda",
    ):
        super().__init__()
        self.model_path = model_path
        self.batch_size = batch_size
        self.device = device

    def _fit_predict(self, task: fev.Task) -> list[datasets.DatasetDict]:
        # Necessary to avoid infinite recursion during custom kernel compilation inside SageMaker
        contextlib.redirect_stdout = lambda _: contextlib.nullcontext()

        if self.device == "cpu":
            os.environ["TIREX_NO_CUDA"] = "True"

        from tirex import ForecastModel, load_model

        model: ForecastModel = load_model(self.model_path, device=self.device)
        predictions_per_window = []
        for window in task.iter_windows():
            past_data, _ = fev.convert_input_data(window, adapter="datasets", as_univariate=True)
            past_data = past_data.with_format("numpy").cast_column(
                "target", datasets.Sequence(datasets.Value("float32"))
            )
            with self._record_inference_time():
                quantiles, means = model.forecast(
                    context=[t for t in past_data["target"]],
                    quantile_levels=task.quantile_levels,
                    prediction_length=task.horizon,
                    batch_size=self.batch_size,
                )
            predictions_dict = {"predictions": means}
            for idx, level in enumerate(task.quantile_levels):
                predictions_dict[str(level)] = quantiles[:, :, idx]
            predictions_per_window.append(
                fev.utils.combine_univariate_predictions_to_multivariate(
                    datasets.Dataset.from_dict(predictions_dict), target_columns=task.target_columns
                )
            )
        return predictions_per_window

import datasets
import numpy as np

import fev


class ChronosModel(fev.ForecastingModel):
    """Chronos-Bolt model from https://github.com/amazon-science/chronos-forecasting."""

    model_name = "chronos"

    def __init__(
        self,
        model_path: str = "amazon/chronos-bolt-base",
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        batch_size: int = 256,
    ):
        super().__init__()
        self.model_path = model_path
        self.device = device
        self.torch_dtype = torch_dtype
        self.batch_size = batch_size

    def _fit_predict(self, task: fev.Task) -> list[datasets.DatasetDict]:
        import torch
        from chronos import BaseChronosPipeline

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        model_path = fev.utils.maybe_cache_from_s3(self.model_path)
        pipeline = BaseChronosPipeline.from_pretrained(
            model_path, device_map=self.device, torch_dtype=self.torch_dtype
        )

        quantile_levels = task.quantile_levels.copy()
        if 0.5 not in quantile_levels:
            quantile_levels.append(0.5)

        predictions_per_window = []
        for window in task.iter_windows():
            past_data, _ = fev.convert_input_data(window, adapter="datasets", as_univariate=True)
            past_data = past_data.with_format("torch").cast_column(
                "target", datasets.Sequence(datasets.Value("float32"))
            )

            quantiles_all = []
            mean_all = []

            with self._record_inference_time():
                for batch in self._batchify(past_data["target"]):
                    quantiles, mean = pipeline.predict_quantiles(
                        batch,
                        prediction_length=task.horizon,
                        limit_prediction_length=False,
                        quantile_levels=quantile_levels,
                    )
                    quantiles_all.append(quantiles.numpy())
                    mean_all.append(mean.numpy())

            quantiles_np = np.concatenate(quantiles_all, axis=0)
            mean_np = np.concatenate(mean_all, axis=0)

            if task.eval_metric in ["MSE", "RMSE", "RMSSE"]:
                point_forecast = mean_np
            else:
                point_forecast = quantiles_np[:, :, quantile_levels.index(0.5)]

            predictions_dict = {"predictions": point_forecast}
            for idx, level in enumerate(task.quantile_levels):
                predictions_dict[str(level)] = quantiles_np[:, :, idx]

            predictions_per_window.append(
                fev.utils.combine_univariate_predictions_to_multivariate(
                    datasets.Dataset.from_dict(predictions_dict), target_columns=task.target_columns
                )
            )

        return predictions_per_window

    def _batchify(self, lst: list):
        for i in range(0, len(lst), self.batch_size):
            yield lst[i : i + self.batch_size]

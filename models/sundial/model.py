from typing import Any, Iterator, Literal

import datasets
import numpy as np

import fev


class SundialModel(fev.ForecastingModel):
    """Sundial model from https://huggingface.co/thuml/sundial-base-128m."""

    model_name = "sundial"

    def __init__(
        self,
        model_path: str = "thuml/sundial-base-128m",
        context_length: int = 2880,
        seed: int = 42,
        device: str = "auto",
        num_samples: int = 20,
    ):
        super().__init__()
        self.model_path = model_path
        self.context_length = context_length
        self.seed = seed
        self.device = device
        self.num_samples = num_samples

    def _fit_predict(self, task: fev.Task) -> list[datasets.DatasetDict]:
        import torch
        import transformers
        from gluonts.transform import LastValueImputation
        from transformers import AutoModelForCausalLM

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map=self.device,
            trust_remote_code=True,
        )

        imputation = LastValueImputation()
        transformers.set_seed(self.seed)

        predictions_per_window = []
        for window in task.iter_windows():
            past_data, _ = fev.convert_input_data(window, adapter="datasets", as_univariate=True)

            def loader():
                for d in past_data.with_format("numpy"):
                    yield (
                        torch.tensor(imputation(d["target"].copy())[-self.context_length :], dtype=torch.float32)
                        .unsqueeze(0)
                        .to(self.device)
                    )

            forecasts = []
            with self._record_inference_time():
                for time_series in loader():
                    samples = model.generate(
                        time_series,
                        max_new_tokens=window.horizon,
                        num_samples=self.num_samples,
                    )

                    point_forecast_type = "mean" if task.eval_metric in ["MSE", "RMSE", "RMSSE"] else "median"
                    if point_forecast_type == "mean":
                        forecast = {"predictions": torch.mean(samples, dim=1).detach().cpu().numpy()}
                    else:
                        forecast = {"predictions": torch.quantile(samples, q=0.5, dim=1).detach().cpu().numpy()}

                    for q in task.quantile_levels:
                        forecast[str(q)] = torch.quantile(samples, q=q, dim=1).detach().cpu().numpy()
                    forecasts.append(forecast)

            predictions = datasets.Dataset.from_dict(
                {
                    k: np.concatenate([f[k] for f in forecasts], axis=0)
                    for k in ["predictions"] + [str(q) for q in task.quantile_levels]
                }
            )
            predictions_per_window.append(
                fev.utils.combine_univariate_predictions_to_multivariate(
                    predictions, target_columns=window.target_columns
                )
            )
        return predictions_per_window

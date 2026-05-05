import math

import datasets
import numpy as np

import fev


class TotoModel(fev.ForecastingModel):
    """Toto model from https://github.com/DataDog/toto."""

    model_name = "toto-1.0"

    def __init__(
        self,
        model_path: str = "Datadog/Toto-Open-Base-1.0",
        max_batch_variate_size: int = 24,
        num_samples: int = 256,
        samples_per_batch: int = 8,
        max_context_length: int = 4096,
        as_univariate: bool = False,
        compile_model: bool = False,
        device: str = "auto",
    ):
        super().__init__()
        self.model_path = model_path
        self.max_batch_variate_size = max_batch_variate_size
        self.num_samples = num_samples
        self.samples_per_batch = samples_per_batch
        self.max_context_length = max_context_length
        self.as_univariate = as_univariate
        self.compile_model = compile_model
        self.device = device

    def _fit_predict(self, task: fev.Task) -> list[datasets.DatasetDict]:
        import pandas as pd
        import torch
        from toto.inference.forecaster import TotoForecaster
        from toto.model.toto import Toto

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        toto = Toto.from_pretrained(self.model_path)
        toto.to(self.device)
        if self.compile_model:
            toto.compile()
        toto_forecaster = TotoForecaster(toto.model)

        predictions_per_window = []
        for window in task.iter_windows():
            predictions = self._predict_window(
                window,
                toto_forecaster=toto_forecaster,
                time_delta_seconds=_freq_to_seconds(task.freq),
                quantile_levels=task.quantile_levels,
                return_mean=task.eval_metric in ["MSE", "RMSE", "RMSSE"],
            )
            predictions_per_window.append(predictions)
        return predictions_per_window

    def _predict_window(
        self,
        window: fev.EvaluationWindow,
        toto_forecaster,
        quantile_levels: list[float],
        time_delta_seconds: float,
        return_mean: bool,
    ) -> datasets.DatasetDict:
        import torch
        from toto.data.util.dataset import MaskedTimeseries

        if self.as_univariate:
            past_data, _ = fev.convert_input_data(window, adapter="datasets", as_univariate=True)
            target_columns = ["target"]
        else:
            past_data, _ = window.get_input_data()
            target_columns = window.target_columns

        past_data_features = past_data.features
        past_data_features.update({col: datasets.Sequence(datasets.Value("float32")) for col in target_columns})
        past_data: datasets.Dataset = past_data.cast(past_data_features)

        num_variates = len(target_columns)
        inputs: list[torch.Tensor] = [
            torch.tensor(np.stack(tuple(row.values()), axis=0), dtype=torch.float32)
            for row in past_data.select_columns(target_columns)
        ]

        forecast_batches = []
        batch_size = max(1, math.floor(self.max_batch_variate_size / num_variates))
        with self._record_inference_time():
            for batch in _batchify(inputs, batch_size=batch_size):
                stacked_batch = _left_pad_and_stack_2d(batch)
                stacked_batch = stacked_batch[..., -self.max_context_length :]
                stacked_batch = stacked_batch.to(device=self.device)
                stacked_batch = _ffill(stacked_batch)
                nan_mask = torch.isnan(stacked_batch)
                stacked_batch[nan_mask] = 0.0

                current_batch_size, _, context_length = stacked_batch.shape
                id_mask = torch.arange(current_batch_size, dtype=torch.int, device=self.device)[:, None, None].repeat(
                    1, num_variates, context_length
                )
                timestamp_seconds = torch.zeros_like(stacked_batch, dtype=torch.int)
                time_interval_seconds = torch.full(
                    (current_batch_size, 1), fill_value=time_delta_seconds, device=self.device, dtype=torch.int
                )

                masked_timeseries = MaskedTimeseries(
                    series=stacked_batch,
                    padding_mask=~nan_mask,
                    id_mask=id_mask,
                    timestamp_seconds=timestamp_seconds,
                    time_interval_seconds=time_interval_seconds,
                )

                toto_forecast = toto_forecaster.forecast(
                    masked_timeseries,
                    prediction_length=window.horizon,
                    num_samples=self.num_samples,
                    samples_per_batch=self.samples_per_batch,
                )

                multivariate_forecast = {variate_name: {} for variate_name in target_columns}
                if return_mean:
                    mean_forecast = toto_forecast.mean.cpu().numpy()
                    for i, variate_name in enumerate(target_columns):
                        multivariate_forecast[variate_name]["predictions"] = mean_forecast[:, i]
                else:
                    median_forecast = toto_forecast.quantile(0.5).cpu().numpy()
                    for i, variate_name in enumerate(target_columns):
                        multivariate_forecast[variate_name]["predictions"] = median_forecast[:, i]

                for q in quantile_levels:
                    q_forecast = toto_forecast.quantile(q).cpu().numpy()
                    for i, variate_name in enumerate(target_columns):
                        multivariate_forecast[variate_name][str(q)] = q_forecast[:, i]
                forecast_batches.append(multivariate_forecast)

        predictions_dict = {}
        for variate_name in target_columns:
            predictions_dict[variate_name] = datasets.Dataset.from_dict(
                {
                    k: np.concatenate([batch[variate_name][k] for batch in forecast_batches], axis=0)
                    for k in ["predictions"] + [str(q) for q in quantile_levels]
                }
            )
        predictions = datasets.DatasetDict(predictions_dict)
        predictions.set_format("numpy")
        if self.as_univariate:
            predictions = fev.utils.combine_univariate_predictions_to_multivariate(predictions, window.target_columns)
        return predictions


def _batchify(lst: list, batch_size: int):
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]


def _ffill(tensor):
    """Forward fill along the last axis."""
    import torch

    nan_mask = torch.isnan(tensor)
    indices = torch.where(nan_mask, 0, torch.arange(tensor.shape[-1], device=tensor.device).expand_as(tensor))
    last_valid = torch.cummax(indices, dim=-1).values
    return torch.gather(tensor, dim=-1, index=last_valid)


def _left_pad_and_stack_2d(tensors):
    import torch

    max_len = max(c.shape[-1] for c in tensors)
    padded = []
    for c in tensors:
        padding = torch.full(size=(c.shape[0], max_len - c.shape[-1]), fill_value=torch.nan, device=c.device)
        padded.append(torch.concat((padding, c), dim=-1))
    return torch.stack(padded)


def _freq_to_seconds(freq) -> float:
    import pandas as pd

    if isinstance(freq, str):
        freq = pd.tseries.frequencies.to_offset(freq)
    try:
        return freq.nanos / 1e9
    except ValueError:
        if isinstance(freq, pd.offsets.BusinessDay):
            return freq.n * 24 * 60 * 60
        elif isinstance(freq, pd.offsets.Week):
            return freq.n * 7 * 24 * 60 * 60
        elif isinstance(freq, (pd.offsets.MonthBegin, pd.offsets.MonthEnd)):
            return 30 * 24 * 60 * 60
        elif isinstance(freq, (pd.offsets.QuarterEnd, pd.offsets.QuarterBegin)):
            return 90 * 24 * 60 * 60
        elif isinstance(freq, (pd.offsets.YearEnd, pd.offsets.YearBegin)):
            return 365.25 * 24 * 60 * 60
        else:
            raise ValueError(f"Cannot handle frequency of type {type(freq)}: {freq}")

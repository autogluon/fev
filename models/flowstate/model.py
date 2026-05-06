"""Granite FlowState-R1 model wrapper."""

import datasets
import numpy as np

import fev


class FlowStateModel(fev.ForecastingModel):
    """FlowState model from https://github.com/ibm-granite/granite-tsfm."""

    model_name = "flowstate"
    # Datasets from GiftEvalPretrain
    trained_on_datasets = [
        "favorita_transactions_1D",
        "fred_md_2025",
        "kdd_cup_2022_10T",
        "m5_1D",
        "proenfo_gfc12",
        "proenfo_gfc14",
        "proenfo_gfc17",
    ]

    def __init__(
        self,
        model_path: str = "ibm-granite/granite-timeseries-flowstate-r1",
        revision: str = "r1.1",
        batch_size: int = 16,
        device: str = "auto",
        seed: int = 0,
    ):
        super().__init__()
        self.model_path = model_path
        self.revision = revision
        self.batch_size = batch_size
        self.device = device
        self.seed = seed

    def _fit_predict(self, task: fev.Task) -> list[datasets.DatasetDict]:
        import torch
        from tsfm_public import FlowStateForPrediction

        torch.manual_seed(self.seed)
        task.load_full_dataset()

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        import warnings

        model_path = fev.utils.maybe_cache_from_s3(self.model_path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = FlowStateForPrediction.from_pretrained(model_path, revision=self.revision).to(self.device)
        model.eval()

        scale_factor = 24.0 / _get_seasonal_period(task.freq, task.seasonality)
        model_quantiles = model.config.quantiles  # [0.1, 0.2, ..., 0.9]
        max_context = int(model.config.context_length / scale_factor)

        predictions_per_window = []
        for window in task.iter_windows():
            predictions = self._predict_window(
                window,
                model=model,
                scale_factor=scale_factor,
                model_quantiles=model_quantiles,
                max_context=max_context,
                quantile_levels=task.quantile_levels,
                return_mean=task.eval_metric in ["MSE", "RMSE", "RMSSE"],
            )
            predictions_per_window.append(predictions)
        return predictions_per_window

    def _predict_window(
        self,
        window: fev.EvaluationWindow,
        model,
        scale_factor: float,
        model_quantiles: list[float],
        max_context: int,
        quantile_levels: list[float],
        return_mean: bool,
    ) -> datasets.DatasetDict:
        import torch

        past_data, _ = fev.convert_input_data(window, adapter="datasets", as_univariate=True)
        past_data = past_data.with_format("numpy")

        all_contexts = [np.array(row["target"][-max_context:], dtype=np.float32) for row in past_data]

        point_forecasts = []
        quantile_forecasts = {str(q): [] for q in quantile_levels}

        with self._record_inference_time():
            for batch_contexts in _dynamic_batchify(all_contexts, self.batch_size, model.config.context_length):
                batch_tensor = _prepare_batch(batch_contexts, max_context, self.device)

                with torch.no_grad():
                    output = model(
                        past_values=batch_tensor,
                        scale_factor=scale_factor,
                        prediction_length=window.horizon,
                        batch_first=True,
                    )

                pred = output.prediction_outputs[:, :, 0].cpu().numpy()
                quants = output.quantile_outputs[:, :, :, 0].cpu().numpy()

                # Enforce non-negative when all past values are non-negative
                for i, ctx in enumerate(batch_contexts):
                    if np.all(np.nan_to_num(ctx, nan=1.0) >= 0):
                        pred[i] = np.maximum(pred[i], 0.0)
                        quants[i] = np.maximum(quants[i], 0.0)

                if return_mean:
                    point_forecasts.append(pred)
                else:
                    median_idx = model_quantiles.index(0.5)
                    point_forecasts.append(quants[:, median_idx])

                for q in quantile_levels:
                    q_idx = _closest_quantile_idx(q, model_quantiles)
                    quantile_forecasts[str(q)].append(quants[:, q_idx])

        predictions_dict = {"predictions": np.concatenate(point_forecasts, axis=0)}
        for q in quantile_levels:
            predictions_dict[str(q)] = np.concatenate(quantile_forecasts[str(q)], axis=0)

        return fev.utils.combine_univariate_predictions_to_multivariate(
            datasets.Dataset.from_dict(predictions_dict),
            target_columns=window.target_columns,
        )


def _dynamic_batchify(contexts: list[np.ndarray], base_batch_size: int, pretrain_context: int):
    """Yield batches with dynamic size — reduce batch size for long contexts, never exceed base."""
    i = 0
    while i < len(contexts):
        ctx_len = len(contexts[i])
        batch_size = min(base_batch_size, max(1, int(base_batch_size * pretrain_context / ctx_len)))
        yield contexts[i : i + batch_size]
        i += batch_size


def _prepare_batch(contexts: list[np.ndarray], max_context: int, device: str):
    """Left-pad, truncate, and ffill NaN values."""
    import torch

    truncated = [ctx[-max_context:] for ctx in contexts]
    max_len = max(len(c) for c in truncated)

    batch = np.full((len(truncated), max_len, 1), np.nan, dtype=np.float32)
    for i, ctx in enumerate(truncated):
        batch[i, max_len - len(ctx) :, 0] = ctx

    tensor = torch.from_numpy(batch).to(device)
    nan_mask = torch.isnan(tensor[:, :, 0])
    if nan_mask.any():
        for i in range(tensor.shape[0]):
            seq = tensor[i, :, 0]
            mask = torch.isnan(seq)
            if mask.any():
                indices = torch.where(
                    mask, torch.zeros_like(mask, dtype=torch.long), torch.arange(len(seq), device=device)
                )
                indices = torch.cummax(indices, dim=0).values
                tensor[i, :, 0] = seq[indices]
                first_valid = (~mask).nonzero(as_tuple=True)[0]
                if len(first_valid) > 0:
                    tensor[i, : first_valid[0], 0] = seq[first_valid[0]]
                else:
                    tensor[i, :, 0] = 0.0
    return tensor


def _closest_quantile_idx(q: float, model_quantiles: list[float]) -> int:
    return min(range(len(model_quantiles)), key=lambda i: abs(model_quantiles[i] - q))


def _get_seasonal_period(freq: str, seasonality: int) -> int:
    """Determine the seasonal period for FlowState's scale_factor.

    Uses task.seasonality when it captures a real cycle (> 1). Falls back to
    an approximate yearly period for daily+, or steps-per-day for sub-daily.
    """
    if seasonality > 1:
        return seasonality

    import pandas as pd

    offset = pd.tseries.frequencies.to_offset(freq)

    if isinstance(offset, (pd.offsets.YearBegin, pd.offsets.YearEnd)):
        return 4
    if isinstance(offset, (pd.offsets.QuarterBegin, pd.offsets.QuarterEnd)):
        return 4
    if isinstance(offset, (pd.offsets.MonthBegin, pd.offsets.MonthEnd)):
        return 12
    if isinstance(offset, pd.offsets.Week):
        return 52
    if isinstance(offset, (pd.offsets.Day, pd.offsets.BusinessDay)):
        return 365
    # Sub-daily: use steps per day
    try:
        nanos_per_day = 24 * 3600 * 10**9
        return max(1, round(nanos_per_day / offset.nanos))
    except (ValueError, AttributeError):
        return 1

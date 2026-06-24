import datasets
import numpy as np

import fev

# Toto 2.0 always returns these nine quantile levels; arbitrary task levels are interpolated from them.
TOTO_QUANTILES = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


class Toto2Model(fev.ForecastingModel):
    """Toto 2.0 model from https://github.com/DataDog/toto (https://pypi.org/project/toto-2)."""

    model_name = "toto-2.0"
    trained_on_datasets = [
        "favorita_transactions_1D",
        "fred_md_2025",
        "proenfo_gfc12",
        "proenfo_gfc14",
        "proenfo_gfc17",
        "kdd_cup_2022_10T",
        "m5_1D",
    ]

    def __init__(
        self,
        model_path: str = "Datadog/Toto-2.0-22m",
        max_batch_variate_size: int = 24,
        max_context_length: int = 4096,
        decode_block_size: int | None = None,
        device: str = "auto",
        seed: int = 42,
    ):
        super().__init__()
        self.model_path = model_path
        self.max_batch_variate_size = max_batch_variate_size
        self.max_context_length = max_context_length
        self.decode_block_size = decode_block_size
        self.device = device
        self.seed = seed

    def _fit_predict(self, task: fev.Task) -> list[datasets.DatasetDict]:
        import torch
        from toto2 import Toto2Model as Toto2

        torch.manual_seed(self.seed)
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        model = Toto2.from_pretrained(fev.utils.maybe_cache_from_s3(self.model_path))
        model = model.to(self.device).eval()

        return [self._predict_window(window, model, task.quantile_levels) for window in task.iter_windows()]

    def _predict_window(
        self,
        window: fev.EvaluationWindow,
        model,
        quantile_levels: list[float],
    ) -> datasets.DatasetDict:
        import torch

        target_columns = window.target_columns
        num_variates = len(target_columns)

        past_data, _ = window.get_input_data()
        past_data = past_data.select_columns(target_columns).cast(
            datasets.Features({col: datasets.Sequence(datasets.Value("float32")) for col in target_columns})
        )
        # One tensor of shape (num_variates, context_length) per time series item.
        series = [torch.tensor(np.stack(list(row.values())), dtype=torch.float32) for row in past_data]

        batch_size = max(1, self.max_batch_variate_size // num_variates)
        forecasts: list[np.ndarray] = []  # each entry: (num_quantiles, batch, num_variates, horizon)
        with self._record_inference_time():
            for batch in _batchify(series, batch_size):
                target, mask = _left_pad_and_stack(batch, self.max_context_length, self.device)
                series_ids = torch.zeros(len(batch), num_variates, dtype=torch.long, device=self.device)

                quantiles = model.forecast(
                    {"target": target, "target_mask": mask, "series_ids": series_ids},
                    horizon=window.horizon,
                    decode_block_size=self.decode_block_size,
                    has_missing_values=not bool(mask.all()),
                )
                forecasts.append(quantiles.cpu().numpy())

        # (num_quantiles, num_items, num_variates, horizon)
        quantiles = np.concatenate(forecasts, axis=1)
        predictions = {
            variate: {"predictions": _interp(quantiles, 0.5)[:, i]} for i, variate in enumerate(target_columns)
        }
        for q in quantile_levels:
            q_forecast = _interp(quantiles, q)
            for i, variate in enumerate(target_columns):
                predictions[variate][str(q)] = q_forecast[:, i]

        result = datasets.DatasetDict(
            {variate: datasets.Dataset.from_dict(preds) for variate, preds in predictions.items()}
        )
        result.set_format("numpy")
        return result


def _batchify(items: list, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def _left_pad_and_stack(series: list, max_context_length: int, device: str):
    """Left-pad a batch of (num_variates, time) tensors to a common length and return (target, mask)."""
    import torch

    series = [s[..., -max_context_length:] for s in series]
    context_length = max(s.shape[-1] for s in series)
    targets, masks = [], []
    for s in series:
        pad = context_length - s.shape[-1]
        nan_mask = torch.isnan(s)
        targets.append(torch.nn.functional.pad(s.nan_to_num(0.0), (pad, 0)))
        masks.append(torch.nn.functional.pad(~nan_mask, (pad, 0)))  # padded and NaN positions are masked out
    return torch.stack(targets).to(device), torch.stack(masks).to(device)


def _interp(quantiles: np.ndarray, level: float) -> np.ndarray:
    """Linearly interpolate a quantile `level` from the model's fixed `TOTO_QUANTILES` (along axis 0)."""
    if level <= TOTO_QUANTILES[0]:
        return quantiles[0]
    if level >= TOTO_QUANTILES[-1]:
        return quantiles[-1]
    hi = int(np.searchsorted(TOTO_QUANTILES, level))
    lo = hi - 1
    weight = (level - TOTO_QUANTILES[lo]) / (TOTO_QUANTILES[hi] - TOTO_QUANTILES[lo])
    return quantiles[lo] * (1 - weight) + quantiles[hi] * weight

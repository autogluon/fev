"""t0-gamma wrapper for fev-bench.

Forecasts targets separately, with cross-inference for tasks without covariates.
"""

from collections.abc import Sequence
from typing import ClassVar

import datasets
import numpy as np
import torch
from t0 import MaskType, T0Forecaster, TimeSeries, VariateType, batch_series

import fev

# Longest history fed to the model.
DEFAULT_MAX_CONTEXT = 8192
DEFAULT_BATCH_SIZE = 32


def cast_as_numeric(*sequences) -> list[np.ndarray]:
    """Convert values to float32; encode strings across all given sequences."""
    try:
        return [np.asarray(s, dtype=np.float32) for s in sequences]
    except (ValueError, TypeError):
        seqs = [[str(v) for v in s] for s in sequences]
        codes = {value: float(idx) for idx, value in enumerate(sorted({v for s in seqs for v in s}))}
        return [np.asarray([codes[v] for v in s], dtype=np.float32) for s in seqs]


def predict_options(mode: str) -> dict[str, bool]:
    """Enable A2 adjustments or disable all adjustments."""
    if mode not in ("none", "a2"):
        raise ValueError(f"unknown robust mode {mode!r}; expected 'none' or 'a2'")
    return dict.fromkeys(
        ("trim_leading_missing", "symmetric_averaging", "clip_nonnegative", "snap_integers"),
        mode == "a2",
    )


def predict_grouped(
    model,
    groups: Sequence[Sequence[np.ndarray]],
    horizon: int,
    quantile_levels: Sequence[float],
    options: dict[str, bool],
) -> list[np.ndarray]:
    """Forecast each group of targets jointly."""
    device = next(model.parameters()).device
    out = []
    for members in groups:
        width = max(len(c) for c in members)
        block = np.full((len(members), width), np.nan, dtype=np.float32)
        for row, context in enumerate(members):
            block[row, width - len(context) :] = context
        context, mask, group_ids = batch_series([block])
        with torch.inference_mode():
            forecast = model.predict(
                context.to(device),
                horizon=horizon,
                quantile_levels=list(quantile_levels),
                mask=mask.to(device),
                group_ids=group_ids.to(device),
                **options,
            )
        out.append(forecast.quantiles.float().cpu().numpy())
    return out


class T0Model(fev.ForecastingModel):
    """Evaluate t0-gamma on fev tasks."""

    model_name = "t0-gamma"
    # Dataset configs from autogluon/fev_datasets seen during pretraining. Empty:
    # this checkpoint's pretraining mix excludes the fev-bench evaluation corpora.
    trained_on_datasets: ClassVar[list[str]] = []

    def __init__(
        self,
        model_path: str = "theforecastingcompany/t0-gamma",
        device: str = "auto",
        max_context_length: int = DEFAULT_MAX_CONTEXT,
        batch_size: int = DEFAULT_BATCH_SIZE,
        use_covariates: bool = True,
        robust: str = "a2",
        cross_inference: int = 64,
    ):
        super().__init__()
        self.robust = robust
        self.cross_inference = cross_inference
        self.model_path = model_path
        self.device = device
        self.max_context_length = max_context_length
        self.batch_size = batch_size
        self.use_covariates = use_covariates
        self._model = None

    def _load_model(self) -> T0Forecaster:
        """Load the checkpoint once."""
        if self._model is None:
            self._model = T0Forecaster.from_pretrained(self.model_path).to(self._resolve_device()).eval()
        return self._model

    def _resolve_device(self) -> str:
        """Use the requested device or detect CUDA, MPS, or CPU."""
        if self.device != "auto":
            return self.device
        if torch.cuda.is_available():
            return "cuda"
        return "mps" if torch.backends.mps.is_available() else "cpu"

    def convert_series(
        self, past_cols, future_cols, index: int, known: list[str], past_only: list[str], horizon: int
    ) -> TimeSeries:
        """Build one target series with its covariates."""
        (target,) = cast_as_numeric(past_cols["target"][index])
        target = target[-self.max_context_length :]
        context_len = len(target)

        future_covariates = None
        if known:
            # A known column spans context and horizon: its past half lives in the
            # window's past_data, its future half in future_data.
            rows = []
            for c in known:
                seen, ahead = cast_as_numeric(past_cols[c][index], future_cols[c][index])
                rows.append(np.concatenate([seen[-context_len:], ahead[:horizon]]))
            future_covariates = torch.from_numpy(np.stack(rows))[None]

        series = TimeSeries.from_array(
            torch.from_numpy(target)[None],
            future_covariates=future_covariates,
            horizon=horizon,
        )
        if not past_only:
            return series

        # Historical rows are known over the context and unknown across the
        # horizon, so their forecast region is MISSING rather than WITHHELD.
        width = series.seq_len
        values = np.stack([cast_as_numeric(past_cols[c][index])[0][-context_len:] for c in past_only])
        hist = torch.zeros((len(past_only), width), dtype=torch.float32)
        hist[:, :context_len] = torch.from_numpy(np.nan_to_num(values, nan=0.0))
        mask = torch.full((len(past_only), width), MaskType.MISSING, dtype=torch.int8)
        mask[:, :context_len] = MaskType.VALID
        # A gap in a covariate is an absent observation, not a zero: left VALID it
        # would feed NaN to the scaler and turn the whole forecast non-finite.
        gaps = torch.from_numpy(np.isnan(values))
        mask[:, :context_len] = torch.where(
            gaps, torch.tensor(MaskType.MISSING, dtype=torch.int8), mask[:, :context_len]
        )
        return TimeSeries(
            variates=torch.cat([series.variates, hist]),
            mask=torch.cat([series.mask, mask]),
            group_ids=torch.cat([series.group_ids, torch.zeros_like(mask, dtype=torch.long)]),
            variate_type=torch.cat(
                [series.variate_type, torch.full_like(mask, VariateType.HISTORICAL, dtype=torch.long)]
            ),
        )

    def _fit_predict(self, task: fev.Task) -> list[datasets.DatasetDict]:
        device = self._resolve_device()
        model = self._load_model()

        known = list(task.known_dynamic_columns or []) if self.use_covariates else []
        past_only = list(task.past_dynamic_columns or []) if self.use_covariates else []

        # fev scores the point forecast with the task's own metric; a squared-error
        # metric wants the mean, everything else the median. The model emits
        # quantiles only, so 0.5 stands in for the mean.
        levels = list(task.quantile_levels)
        point_levels = sorted(levels if 0.5 in levels else [*levels, 0.5])

        task.load_full_dataset()  # task.freq reads the loaded data
        options = predict_options(self.robust)
        grouped = self.cross_inference > 1 and not (known or past_only)

        predictions_per_window = []
        for window in task.iter_windows():
            past_data, future_data = fev.convert_input_data(window, adapter="datasets", as_univariate=True)
            n_series = len(past_data)
            # Read each column once. Indexing an Arrow dataset row-by-row inside the
            # batch loop re-materialises every column of that row each time.
            past_cols = {c: past_data[c] for c in ["target", *known, *past_only]}
            future_cols = {c: future_data[c] for c in known} if future_data is not None else {}

            quantiles_all = []
            with self._record_inference_time():
                if grouped:
                    contexts = [
                        cast_as_numeric(past_cols["target"][i])[0][-self.max_context_length :] for i in range(n_series)
                    ]
                    groups = [contexts[s : s + self.cross_inference] for s in range(0, n_series, self.cross_inference)]
                    quantiles_all = predict_grouped(model, groups, task.horizon, point_levels, options)
                else:
                    rows_per_series = 1 + len(known) + len(past_only)
                    series_per_batch = max(1, self.batch_size // rows_per_series)
                    for start in range(0, n_series, series_per_batch):
                        stop = min(start + series_per_batch, n_series)
                        if known or past_only:
                            chunk = [
                                self.convert_series(past_cols, future_cols, i, known, past_only, task.horizon)
                                for i in range(start, stop)
                            ]
                            model_input = TimeSeries.batch(chunk).to(device)
                            out = self._predict(
                                model, model_input, task.horizon, point_levels, batched=True, **options
                            )
                        else:
                            contexts = [
                                cast_as_numeric(past_cols["target"][i])[0][-self.max_context_length :]
                                for i in range(start, stop)
                            ]
                            context, mask, group_ids = batch_series(contexts)
                            out = self._predict(
                                model,
                                context.to(device),
                                task.horizon,
                                point_levels,
                                mask=mask.to(device),
                                group_ids=group_ids.to(device),
                                **options,
                            )
                        quantiles_all.append(out)

            quantiles_np = np.concatenate(quantiles_all, axis=0)  # (series, horizon, quantiles)

            predictions_dict = {"predictions": quantiles_np[:, :, point_levels.index(0.5)]}
            for level in task.quantile_levels:
                predictions_dict[str(level)] = quantiles_np[:, :, point_levels.index(level)]

            predictions_per_window.append(
                fev.combine_univariate_predictions_to_multivariate(
                    datasets.Dataset.from_dict(predictions_dict), target_columns=task.target_columns
                )
            )

        return predictions_per_window

    def _predict(self, model, model_input, horizon, levels, *, batched=False, **kwargs) -> np.ndarray:
        with torch.inference_mode():
            if batched:
                kwargs["context_length"] = model_input.seq_len - horizon
            out = model.predict(model_input, horizon=horizon, quantile_levels=levels, **kwargs)
        return out.quantiles.float().cpu().numpy()

"""t0-beta model wrapper for fev evaluation.

t0 is a decoder-style patch transformer from The Forecasting Company that emits
quantile forecasts and natively conditions on covariates. `t0-beta` is its
256M-parameter second public checkpoint.

- Weights: https://huggingface.co/theforecastingcompany/t0-beta
- Runtime: https://pypi.org/project/tfc-t0/

A task's columns map onto t0 variates of one sample, which attend to one another
through the model's group attention: every target column is a TARGET row and all
are forecast in one pass, ``known_dynamic_columns`` span context and horizon as
FUTURE rows, and ``past_dynamic_columns`` stop at the forecast start as
HISTORICAL rows. ``use_covariates=False`` drops the covariate rows and
``as_univariate=True`` forecasts each target column on its own, for ablations.

Usage:
    python models/evaluate.py -m t0-beta
"""

import datasets
import numpy as np
import torch
from t0 import MaskType, T0Forecaster, TimeSeries, VariateType, batch_series

import fev

# Longest history fed to the model; longer series keep their most recent steps.
DEFAULT_MAX_CONTEXT = 8192
DEFAULT_BATCH_SIZE = 32


def cast_as_numeric(*sequences) -> list[np.ndarray]:
    """Coerce covariate values to float32, label-encoding non-numeric ones."""
    try:
        return [np.asarray(s, dtype=np.float32) for s in sequences]
    except (ValueError, TypeError):
        seqs = [[str(v) for v in s] for s in sequences]
        codes = {value: float(idx) for idx, value in enumerate(sorted({v for s in seqs for v in s}))}
        return [np.asarray([codes[v] for v in s], dtype=np.float32) for s in seqs]


class T0BetaModel(fev.ForecastingModel):
    """t0-beta forecasting foundation model from The Forecasting Company."""

    model_name = "t0-beta"
    # t0-beta was trained with no data leakage.
    trained_on_datasets = []

    def __init__(
        self,
        model_path: str = "theforecastingcompany/t0-beta",
        device: str = "auto",
        max_context_length: int = DEFAULT_MAX_CONTEXT,
        batch_size: int = DEFAULT_BATCH_SIZE,
        use_covariates: bool = True,
        as_univariate: bool = False,
    ):
        super().__init__()
        self.model_path = model_path
        self.device = device
        self.max_context_length = max_context_length
        self.batch_size = batch_size
        self.use_covariates = use_covariates
        self.as_univariate = as_univariate
        self._model = None

    def _load_model(self) -> T0Forecaster:
        """Load the forecaster once and reuse it across tasks."""
        if self._model is None:
            model_path = fev.utils.maybe_cache_from_s3(self.model_path)
            self._model = T0Forecaster.from_pretrained(model_path).to(self._resolve_device()).eval()
        return self._model

    def _resolve_device(self) -> str:
        """Honour an explicit device, else take the best of cuda, mps, cpu."""
        if self.device != "auto":
            return self.device
        if torch.cuda.is_available():
            return "cuda"
        return "mps" if torch.backends.mps.is_available() else "cpu"

    def convert_series(
        self,
        past_cols,
        future_cols,
        index: int,
        targets: list[str],
        known: list[str],
        past_only: list[str],
        horizon: int,
    ) -> TimeSeries:
        """One series as a t0 ``TimeSeries``: target, known-future and historical rows.

        Every target column becomes a TARGET row of the same sample, so the columns
        attend to one another and are forecast in a single pass.
        """
        context = np.stack([cast_as_numeric(past_cols[c][index])[0][-self.max_context_length :] for c in targets])
        context_len = context.shape[1]

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
            torch.from_numpy(context)[None],  # (1, targets, context)
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
        # as_univariate splits a multivariate task into one instance per target column,
        # so the model then sees a single target row and the columns cannot attend.
        targets = ["target"] if self.as_univariate else list(task.target_columns)

        # fev scores the point forecast with the task's own metric; a squared-error
        # metric wants the mean, everything else the median. The model emits
        # quantiles only, so 0.5 stands in for the mean.
        levels = list(task.quantile_levels)
        point_levels = sorted(levels if 0.5 in levels else [*levels, 0.5])

        predictions_per_window = []
        for window in task.iter_windows():
            past_data, future_data = fev.convert_input_data(
                window, adapter="datasets", as_univariate=self.as_univariate
            )
            n_series = len(past_data)
            # Read each column once. Indexing an Arrow dataset row-by-row inside the
            # batch loop re-materialises every column of that row each time.
            past_cols = {c: past_data[c] for c in [*targets, *known, *past_only]}
            future_cols = {c: future_data[c] for c in known} if future_data is not None else {}

            quantiles_all = []
            with self._record_inference_time():
                rows_per_series = len(targets) + len(known) + len(past_only)
                series_per_batch = max(1, self.batch_size // rows_per_series)
                for start in range(0, n_series, series_per_batch):
                    stop = min(start + series_per_batch, n_series)
                    if known or past_only:
                        chunk = [
                            self.convert_series(past_cols, future_cols, i, targets, known, past_only, task.horizon)
                            for i in range(start, stop)
                        ]
                        model_input = TimeSeries.batch(chunk).to(device)
                        out = self._predict(model, model_input, task.horizon, point_levels, batched=True)
                    else:
                        # One (targets, context) entry per series: batch_series gives its rows a
                        # shared group id, which is what makes the columns attend to one another.
                        contexts = [
                            np.stack(
                                [cast_as_numeric(past_cols[c][i])[0][-self.max_context_length :] for c in targets]
                            )
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
                        )
                    quantiles_all.append(out)

            # Target rows come back in input order, so the series axis splits off cleanly.
            quantiles_np = np.concatenate(quantiles_all, axis=0).reshape(
                n_series, len(targets), task.horizon, len(point_levels)
            )

            per_target = {}
            for position, column in enumerate(targets):
                predictions_dict = {"predictions": quantiles_np[:, position, :, point_levels.index(0.5)]}
                for level in task.quantile_levels:
                    predictions_dict[str(level)] = quantiles_np[:, position, :, point_levels.index(level)]
                per_target[column] = datasets.Dataset.from_dict(predictions_dict)

            if self.as_univariate:
                # Each target column was its own instance; fev knows how to fold them back.
                predictions_per_window.append(
                    fev.combine_univariate_predictions_to_multivariate(
                        per_target["target"], target_columns=task.target_columns
                    )
                )
            else:
                predictions_per_window.append(datasets.DatasetDict(per_target))

        return predictions_per_window

    def _predict(self, model, model_input, horizon, levels, *, batched=False, **kwargs) -> np.ndarray:
        with torch.inference_mode():
            if batched:
                kwargs["context_length"] = model_input.seq_len - horizon
            out = model.predict(model_input, horizon=horizon, quantile_levels=levels, **kwargs)
        return out.quantiles.float().cpu().numpy()

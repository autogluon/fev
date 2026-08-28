import math
import os
from typing import List, Optional, Tuple

import datasets
import numpy as np

import fev


class MissingValueImputation:
    def __call__(self, values: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class DummyValueImputation(MissingValueImputation):
    def __init__(self, dummy_value: float = 0.0) -> None:
        self.dummy_value = dummy_value

    def __call__(self, values: np.ndarray) -> np.ndarray:
        nan_indices = np.where(np.isnan(values))
        values[nan_indices] = self.dummy_value
        return values


class LastValueImputation(MissingValueImputation):
    def __call__(self, values: np.ndarray) -> np.ndarray:
        if len(values) == 1 or np.isnan(values).all():
            return DummyValueImputation()(values)
        values = np.expand_dims(values, axis=0)
        mask = np.isnan(values)
        idx = np.where(~mask, np.arange(mask.shape[1]), 0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        out = values[np.arange(idx.shape[0])[:, None], idx]
        values = np.squeeze(out)
        mask = np.isnan(values)
        values[mask] = np.interp(
            np.flatnonzero(mask), np.flatnonzero(~mask), values[~mask]
        )
        return values


def preprocess_string_columns(
    dataset: datasets.Dataset, columns: List[str], bucket_size: int = 100
) -> datasets.Dataset:
    cols_to_hash = []
    for col in columns:
        if col not in dataset.features:
            continue
        feat = dataset.features[col]
        if hasattr(feat, "dtype") and feat.dtype == "string":
            cols_to_hash.append(col)
        elif (
            hasattr(feat, "feature")
            and hasattr(feat.feature, "dtype")
            and feat.feature.dtype == "string"
        ):
            cols_to_hash.append(col)
    if not cols_to_hash:
        return dataset

    def _hash_batch(batch):
        updates = {}
        for col in cols_to_hash:
            hashed_col = []
            for seq in batch[col]:
                hashed_seq = [
                    float(abs(hash(x)) % bucket_size) if isinstance(x, str) else 0.0
                    for x in seq
                ]
                hashed_col.append(hashed_seq)
            updates[col] = hashed_col
        return updates

    return dataset.map(_hash_batch, batched=True)


def prepare_covariates_for_timesfm3(
    task: fev.Task,
    imputation: MissingValueImputation,
    past_batch: dict,
    future_batch: Optional[dict],
    max_context_length: int = 15360,
    target_columns: Optional[List[str]] = None,
) -> Tuple[List[np.ndarray], List[Optional[np.ndarray]], List[Optional[np.ndarray]]]:
    if target_columns is None:
        target_columns = task.target_columns
    if isinstance(target_columns, str):
        target_columns = [target_columns]

    batch_size = len(past_batch[target_columns[0]])
    inputs = []
    batch_pf_covariates = []
    batch_po_covariates = []

    for i in range(batch_size):
        pf_covs_list = []
        po_covs_list = []
        target_list = []
        orig_context_length = None

        for col in target_columns:
            b = np.array(past_batch[col][i], dtype=np.float32)
            inp = imputation(b.copy()) if np.isnan(b).any() else b
            if orig_context_length is None:
                orig_context_length = len(inp)
            if len(inp) > max_context_length:
                inp = inp[-max_context_length:]
            target_list.append(inp)
        inputs.append(np.stack(target_list, axis=0))

        if future_batch is not None:
            for col in task.known_dynamic_columns:
                p_data = np.array(past_batch[col][i], dtype=np.float32)
                f_data = np.array(future_batch[col][i], dtype=np.float32)
                p_imp = imputation(p_data.copy()) if np.isnan(p_data).any() else p_data
                f_imp = imputation(f_data.copy()) if np.isnan(f_data).any() else f_data
                if len(p_imp) > max_context_length:
                    p_imp = p_imp[-max_context_length:]
                if (
                    np.isnan(p_imp).sum() / p_imp.size > 0.25
                    or np.isnan(f_imp).sum() / f_imp.size > 0.25
                ):
                    continue
                full_cov = np.concatenate([p_imp, f_imp], axis=-1)
                pf_covs_list.append(full_cov)

        for col in task.past_dynamic_columns:
            p_data = np.array(past_batch[col][i], dtype=np.float32)
            p_imp = imputation(p_data.copy()) if np.isnan(p_data).any() else p_data
            if len(p_imp) > max_context_length:
                p_imp = p_imp[-max_context_length:]
            if np.isnan(p_imp).sum() / p_imp.size > 0.25:
                continue
            po_covs_list.append(p_imp)

        if pf_covs_list:
            batch_pf_covariates.append(np.stack(pf_covs_list, axis=0))
        else:
            batch_pf_covariates.append(None)

        if po_covs_list:
            batch_po_covariates.append(np.stack(po_covs_list, axis=0))
        else:
            batch_po_covariates.append(None)

    return inputs, batch_pf_covariates, batch_po_covariates


def _batchify(lst1, lst2, batch_size):
    for i in range(0, len(lst1), batch_size):
        b1 = lst1.select(range(i, min(i + batch_size, len(lst1))))
        b2 = lst2.select(range(i, min(i + batch_size, len(lst2)))) if lst2 is not None else None
        yield b1, b2


class TimesFM3Model(fev.ForecastingModel):
    """TimesFM-3 model wrapper for AutoGluon fev-bench with multivariate targets and dynamic covariates."""

    model_name = "timesfm-3"
    trained_on_datasets = []
    TIMESFM_MODEL_QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    def __init__(
        self,
        checkpoint_path: str = "google/timesfm-3.0-pytorch",
        min_batch: int = 4,
        max_batch: int = 64,
        per_core_batch_size: int = 64,
        max_context_length: int = 15360,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.min_batch = min_batch
        self.max_batch = max_batch
        self.per_core_batch_size = per_core_batch_size
        self.max_context_length = max_context_length
        self.device = device
        self._forecaster = None

    def _get_forecaster(self):
        if self._forecaster is None:
            import torch
            from timesfm3 import TimesFM3Evaluator, ModelConfig

            device = self.device
            if device is None or device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"

            checkpoint = fev.utils.maybe_cache_from_s3(self.checkpoint_path)
            config = ModelConfig(
                checkpoint_path=checkpoint,
                per_core_batch_size=self.per_core_batch_size,
                use_variate_attention=True,
                use_sdpa=True,
                device=device,
            )
            self._forecaster = TimesFM3Evaluator(config)
        return self._forecaster

    @staticmethod
    def get_optimal_batch_size(
        max_context_len: int,
        num_variates: int = 1,
        min_batch: int = 4,
        max_batch: int = 64,
    ) -> int:
        """Compute dynamic batch size scaled by total variates (capped at 32 chunk limit) and context length, rounded to nearest power of 2."""
        full_context_count_per_ts = min(32, num_variates) * min(1.0, max(max_context_len, 32) / 15360.0)
        batch_size = 64.0 / max(full_context_count_per_ts, 0.01)
        power_of_2 = int(2 ** round(math.log2(batch_size))) if batch_size > 0 else 1
        return int(np.clip(power_of_2, min_batch, max_batch))

    def _fit_predict(self, task: fev.Task) -> list[datasets.DatasetDict]:
        import torch

        forecaster = self._get_forecaster()
        predictions_per_window = []
        imputation = LastValueImputation()
        windows = list(task.iter_windows())

        for w_idx, window in enumerate(windows):
            past_data, future_data = window.get_input_data()
            all_dynamic = task.known_dynamic_columns + task.past_dynamic_columns
            past_data = preprocess_string_columns(past_data, all_dynamic)
            if future_data is not None:
                future_data = preprocess_string_columns(future_data, task.known_dynamic_columns)

            past_features = past_data.features.copy()
            past_features.update({
                col: datasets.Sequence(datasets.Value("float32"))
                for col in task.known_dynamic_columns + task.target_columns + task.past_dynamic_columns
                if col in past_data.features
            })
            past_data = past_data.cast(past_features)
            if future_data is not None:
                future_features = future_data.features.copy()
                future_features.update({
                    col: datasets.Sequence(datasets.Value("float32"))
                    for col in task.known_dynamic_columns
                    if col in future_data.features
                })
                future_data = future_data.cast(future_features)

            total_series = len(past_data)
            first_target_col = task.target_columns[0]
            max_context_len = max((len(row) for row in past_data[first_target_col]), default=32)
            total_variates = len(task.target_columns) + len(task.known_dynamic_columns) + len(task.past_dynamic_columns)

            dynamic_batch_size = self.get_optimal_batch_size(
                max_context_len=max_context_len,
                num_variates=total_variates,
                min_batch=self.min_batch,
                max_batch=self.max_batch,
            )

            quantile_to_index = {
                q: int(np.argmin(np.abs(np.array(self.TIMESFM_MODEL_QUANTILES) - q)))
                for q in task.quantile_levels
            }
            forecast_batches = []
            with self._record_inference_time():
                for past_batch, future_batch in _batchify(past_data, future_data, dynamic_batch_size):
                    inputs, batch_pf, batch_po = prepare_covariates_for_timesfm3(
                        task,
                        imputation,
                        past_batch,
                        future_batch,
                        max_context_length=self.max_context_length,
                        target_columns=task.target_columns,
                    )
                    batch_outs = list(forecaster.predict_batch(
                        contexts=inputs,
                        horizon=window.horizon,
                        past_only_covariates=batch_po,
                        past_future_covariates=batch_pf,
                        return_quantiles=True,
                        use_symmetric_averaging=True,
                        make_positive=True,
                        sort_quantiles=True,
                    ))

                    for out in batch_outs:
                        f = out.forecast
                        q = out.quantiles
                        if f.ndim == 1:
                            f = f[np.newaxis, :]
                        if q.ndim == 2:
                            q = q[np.newaxis, ...]

                        b_dict = {"predictions": f[:len(task.target_columns), :window.horizon]}
                        for q_lvl in task.quantile_levels:
                            q_i = quantile_to_index[q_lvl]
                            b_dict[str(q_lvl)] = q[:len(task.target_columns), :window.horizon, q_i]
                        forecast_batches.append(b_dict)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

            predictions_dict = {}
            predictions_dict["predictions"] = np.concatenate([b["predictions"] for b in forecast_batches], axis=0)
            for q_lvl in task.quantile_levels:
                predictions_dict[str(q_lvl)] = np.concatenate([b[str(q_lvl)] for b in forecast_batches], axis=0)

            predictions = datasets.Dataset.from_dict(predictions_dict)
            window_pred = fev.combine_univariate_predictions_to_multivariate(
                predictions, target_columns=task.target_columns
            )
            predictions_per_window.append(window_pred)

        return predictions_per_window

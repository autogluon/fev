"""citras-fm model wrapper for fev evaluation.

Loads CITRAS-FM from the Hugging Face Hub (hitachi-nlp/citras-fm) and runs
zero-shot multivariate / covariate-aware forecasting.

Usage:
    python models/evaluate.py -m citras-fm
"""

from __future__ import annotations

import datasets
import numpy as np
import torch

import fev


_DEFAULT_REPO = "hitachi-nlp/citras-fm"
_DEFAULT_REVISION = "ckpt-2026-07-08"


def _to_float32(arr) -> np.ndarray:
    """Convert array-like (numpy, list, pandas, pyarrow) to float32 ndarray."""
    if hasattr(arr, "to_pylist"):
        arr = arr.to_pylist()
    elif hasattr(arr, "as_py"):
        arr = arr.as_py()
    return np.asarray(arr, dtype=np.float32)


def _format2d(seq: np.ndarray) -> np.ndarray:
    """Ensure array is 2-D [C, L].  1-D inputs become [1, L]."""
    return seq if seq.ndim == 2 else seq.reshape(1, -1)


class CitrasFMModel(fev.ForecastingModel):
    """CITRAS-FM: Covariate-Informed Transformer for Time Series Foundation Modeling.

    Zero-shot foundation model that supports multivariate targets and
    past / known covariates natively.

    Args:
        model_path:        Hugging Face Hub repo id (default ``hitachi-nlp/citras-fm``)
                           or a local directory containing ``model.safetensors``
                           and ``config.json``.
        device:            ``"cuda"``, ``"cpu"``, or ``"auto"``.
        batch_size:        Number of series processed per forward pass.
        ignore_covariates: When True, all covariates are discarded (ablation mode).
    """

    model_name = "citras-fm"
    trained_on_datasets = []  # No overlap with any datasets in fev.

    def __init__(
        self,
        model_path: str = _DEFAULT_REPO,
        device: str = "auto",
        batch_size: int = 64,
        ignore_covariates: bool = False,
    ):
        super().__init__()
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size
        self.ignore_covariates = ignore_covariates
        self._model = None  # lazy-loaded

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_device(self) -> str:
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    def _get_model(self):
        if self._model is None:
            from citras_fm import CitrasFM

            device = self._resolve_device()
            print(f"[citras-fm] Loading {self.model_path} (revision={_DEFAULT_REVISION})")
            self._model = CitrasFM.from_pretrained(
                self.model_path,
                revision=_DEFAULT_REVISION,
                map_location=device,
            ).to(device)
            self._model.eval()
        return self._model

    # ------------------------------------------------------------------
    # Batch inference
    # ------------------------------------------------------------------

    def _predict_batch(
        self,
        entries: list[dict],
        horizon: int,
        model,
        device: str,
    ) -> np.ndarray:
        """Run one batch of GluonTS-formatted entries through CITRAS-FM.

        Each entry is expected to contain:
            "target"                  : np.ndarray [Ct, L] or [L]
            "past_feat_dynamic_real"  : np.ndarray [Cp, L]   (optional)
            "feat_dynamic_real"       : np.ndarray [Cf, L+H] (optional)

        Returns:
            np.ndarray of shape [B, H, Ct, Q]
        """
        targets, past_feats, feats = [], [], []
        max_len = 0

        for entry in entries:
            t = _format2d(_to_float32(entry["target"]))             # [Ct, L]
            L = t.shape[1]
            p = _format2d(_to_float32(
                entry.get("past_feat_dynamic_real", np.empty((0, L), dtype=np.float32))
            ))                                                       # [Cp, L]
            f = _format2d(_to_float32(
                entry.get("feat_dynamic_real", np.empty((0, L + horizon), dtype=np.float32))
            ))                                                       # [Cf, L+H]
            targets.append(t)
            past_feats.append(p)
            feats.append(f)
            max_len = max(max_len, L)

        # Left-pad shorter sequences with NaN so the batch is rectangular.
        for i in range(len(targets)):
            pad = max_len - targets[i].shape[1]
            if pad > 0:
                targets[i] = np.concatenate(
                    [np.full((targets[i].shape[0], pad), np.nan, dtype=np.float32), targets[i]], axis=1
                )
                past_feats[i] = np.concatenate(
                    [np.full((past_feats[i].shape[0], pad), np.nan, dtype=np.float32), past_feats[i]], axis=1
                )
                feats[i] = np.concatenate(
                    [np.full((feats[i].shape[0], pad), np.nan, dtype=np.float32), feats[i]], axis=1
                )

        # Stack into batch tensors: [B, C, L] -> transpose -> [B, L, C]
        target_t = torch.from_numpy(np.stack(targets)).transpose(1, 2).to(device)   # [B, L, Ct]
        past_t   = torch.from_numpy(np.stack(past_feats)).transpose(1, 2).to(device) # [B, L, Cp]
        feat_t   = torch.from_numpy(np.stack(feats)).transpose(1, 2).to(device)     # [B, L+H, Cf]

        if self.ignore_covariates:
            B, L, _ = target_t.shape
            past_t = torch.empty((B, L, 0), device=device)
            feat_t = torch.empty((B, L + horizon, 0), device=device)

        _, pred = model.forecast_batch(
            target_t, horizon=horizon, observed_cov=past_t, known_cov=feat_t
        )  # [B, H, Ct, Q]
        return pred.cpu().numpy()

    # ------------------------------------------------------------------
    # fev.ForecastingModel interface
    # ------------------------------------------------------------------

    def _fit_predict(self, task: fev.Task) -> list[datasets.DatasetDict]:
        model = self._get_model()
        device = self._resolve_device()

        q_list: list = model.quantiles
        q_keys = [str(q) for q in q_list]
        median_idx = q_list.index(0.5)

        predictions_per_window: list[datasets.DatasetDict] = []

        for window in task.iter_windows():
            _, pred_ds = fev.convert_input_data(window, adapter="gluonts", as_univariate=False)
            entries = list(pred_ds)

            all_preds: list[np.ndarray] = []
            with self._record_inference_time():
                for start in range(0, len(entries), self.batch_size):
                    batch = entries[start : start + self.batch_size]
                    all_preds.append(
                        self._predict_batch(batch, task.horizon, model, device)
                    )

            pred_arr = np.concatenate(all_preds, axis=0)  # [N, H, Ct, Q]
            pred_t = pred_arr.transpose(0, 3, 1, 2)        # [N, Q, H, Ct]

            predictions_dict: dict[str, datasets.Dataset] = {}
            for col_idx, col in enumerate(task.target_columns):
                sub: dict[str, np.ndarray] = {
                    "predictions": pred_t[:, median_idx, :, col_idx],
                }
                for q in task.quantile_levels:
                    q_str = str(q)
                    sub[q_str] = pred_t[:, q_keys.index(q_str), :, col_idx]
                col_ds = datasets.Dataset.from_dict(sub)
                col_ds.set_format("numpy")
                predictions_dict[col] = col_ds

            predictions_per_window.append(datasets.DatasetDict(predictions_dict))

        return predictions_per_window

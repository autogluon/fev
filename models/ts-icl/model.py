"""
TS-ICL was evaluated on a NVIDIA H100 GPU (92GB) with CUDA version 12.8 installed. 
"""
from pathlib import Path
import sys
from typing import cast, Dict

import datasets
from einops import rearrange
import fev
import numpy as np
import torch

from tsicl import TSICL


class TSICLModel(fev.ForecastingModel):

    """TS-ICL model from https://github.com/EDF-Lab/ts-icl."""
    
    model_name = "ts-icl"
    trained_on_datasets = []

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str = "cuda",
        as_univariate: bool = False,
        batch_size: int = 32
    ) -> None:

        super().__init__()
        self.model_path = model_path
        self.device = device
        self.as_univariate = as_univariate
        self.batch_size = batch_size
        self._model: TSICL | None = None


    def _load_model(self) -> TSICL:
        if self._model is None:
            self._model = TSICL(
                model_path          = self.model_path,
                checkpoint_version  = "tsicl-v1.ckpt",
                allow_auto_download = True,
            )
        return self._model

    
    def _fit_predict(self, task: fev.Task) -> list[datasets.DatasetDict]:

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        quantile_levels = task.quantile_levels

        model = self._load_model()

        predictions_per_window = []

        with self._record_inference_time():

            for window in task.iter_windows():

                # prepare data:
                inputs, target_columns, past_dynamic_columns, known_dynamic_columns = (
                    convert_fev_window_to_list_of_dicts_input(
                        window=window,
                        as_univariate=self.as_univariate
                    )
                )

                # run forward pass on every input in the window:
                quantiles_np, means_np = _predict_window(
                    model                 = model,
                    inputs                = inputs,
                    horizon               = task.horizon,
                    quantile_levels       = quantile_levels,
                    max_context_length    = 4096,
                    batch_size            = self.batch_size,
                    device_map            = self.device
                )

                # store forecast in the right format (from chronos2 pipeline):
                multivariate_forecast: dict[str, dict[str, np.ndarray]] = {
                    variate_name: {} for variate_name in target_columns
                }
                point_forecast = means_np  # [num_items, n_variates, horizon]

                for v_idx, variate_name in enumerate(target_columns):
                    multivariate_forecast[variate_name]["predictions"] = point_forecast[:, v_idx]

                for q_idx, level in enumerate(quantile_levels):
                    for v_idx, variate_name in enumerate(target_columns):
                        multivariate_forecast[variate_name][str(level)] = quantiles_np[:, v_idx, :, q_idx]

                predictions_dict: dict = {}
                for variate_name in target_columns:
                    predictions_dict[variate_name] = datasets.Dataset.from_dict(
                        {
                            k: multivariate_forecast[variate_name][k]
                            for k in ["predictions"] + [str(q) for q in quantile_levels]
                        }
                    )
                predictions = datasets.DatasetDict(predictions_dict)
                predictions.set_format("numpy")

                if self.as_univariate:
                    predictions = fev.utils.combine_univariate_predictions_to_multivariate(
                        predictions, window.target_columns
                    )

                predictions_per_window.append(predictions)
              
        return predictions_per_window


# ---------------------------------------------------------------------------
# TS-ICL utils
# ---------------------------------------------------------------------------


def _predict_window(
    model: TSICL,
    inputs: list,
    horizon: int,
    quantile_levels: list[float],
    max_context_length: int,
    batch_size: int = 64,
    device_map: str = "cuda"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return model predictions for a single FEV window.

    Parameters
    ----------
    model : TSICL
        TSICL model with loaded weights

    inputs : list
        Inputs to forecast, as a list of dict
    
    horizon : int
        Prediction length

    quantile_levels : list[float]
        Quantiles that must be predicted. List of floats between 0 and 1
    
    max_context_length : int
        Maximum context length to be used by TS-ICL (max 4096)

    batch_size : int
        Batch size for processing inputs
    
    device_map : str
        Cpu or cuda device
    
    Returns
    -------
    quantiles_np : np.ndarray
        Predicted quantiles of shape `(N, C, H, Q)`

    mean_np : np.ndarray
        Predicted pointwise estimator of shape `(N, C, H)`
    """
    
    def _nan_var_to_num(
        x: Dict[str, torch.Tensor],
        mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        if mask.sum() > 0:
            x["target"][:,mask] = 0.0
        return x
    
    # replace targets that are only NaNs by zeroes:
    mask = [((~torch.Tensor(x["target"]).isnan()).sum(0) == 0) for x in inputs]
    batch_clean = [_nan_var_to_num(x,m) for x,m in zip(inputs, mask)]

    # model inference:
    try:
        point_q, batch_q = model.forecast(
                inputs            = batch_clean,
                prediction_length = horizon,
                batch_size        = batch_size,
                quantile_levels   = quantile_levels,
                context_length    = max_context_length,
                device            = torch.device(device_map),
                point_estimator   = "median",
                denormalize       = True,
                squeeze_output    = False
        )

        if isinstance(batch_q, list):
            batch_q = torch.stack(batch_q, dim=0) # (b c t q)
        if isinstance(point_q, list):
            point_q = torch.stack(point_q, dim=0) # (b c t 1)
        assert isinstance(batch_q, torch.Tensor) and isinstance(point_q, torch.Tensor)

        # check FEV compliance:
        assert not batch_q.isnan().any()
        assert not point_q.isnan().any()

        quantiles_np = batch_q.cpu().numpy()        # (bs, num_variates, horizon, num_quantiles)
        mean_np = point_q.squeeze(-1).cpu().numpy() # (bs, num_variates, horizon)

    finally:
        sys.stderr = sys.stderr

    return quantiles_np, mean_np


# ---------------------------------------------------------------------------
# fev-bench data helpers
# ---------------------------------------------------------------------------


def _cast_fev_features(
    past_data: "datasets.Dataset",
    future_data: "datasets.Dataset",
    target_columns: list[str],
    past_dynamic_columns: list[str],
    known_dynamic_columns: list[str],
) -> tuple["datasets.Dataset", "datasets.Dataset"]:

    dynamic_columns = [*past_dynamic_columns, *known_dynamic_columns]
    cat_cols = []
    for col in dynamic_columns:
        item = past_data[0][col]
        if not np.issubdtype(item.dtype, np.number):
            cat_cols.append(col)

    numeric_cols = target_columns + list(set(dynamic_columns) - set(cat_cols))
    past_feature_updates = {
        col: datasets.Sequence(datasets.Value("float64")) for col in numeric_cols
    } | {
        col: datasets.Sequence(datasets.Value("string")) for col in cat_cols
    }
    past_data_features = past_data.features
    past_data_features.update(past_feature_updates)
    past_data = past_data.cast(past_data_features)

    future_cat_cols = [k for k in cat_cols if k in known_dynamic_columns]
    future_numeric_cols = list(set(known_dynamic_columns) - set(future_cat_cols))
    future_feature_updates = {col: datasets.Sequence(datasets.Value("float64")) for col in future_numeric_cols} | {
        col: datasets.Sequence(datasets.Value("string")) for col in future_cat_cols
    }
    future_data_features = future_data.features
    future_data_features.update(future_feature_updates)
    future_data = future_data.cast(future_data_features)

    return past_data, future_data


def convert_fev_window_to_list_of_dicts_input(
    window: "fev.EvaluationWindow",
    as_univariate: bool
) -> tuple[list[dict[str, torch.Tensor | dict[str, torch.Tensor]]], list[str], list[str], list[str]]:

    if as_univariate:
        past_data, future_data = fev.convert_input_data(
            window, adapter="datasets", as_univariate=True
        )
        target_columns = ["target"]
        past_dynamic_columns = []
        known_dynamic_columns = []
    else:
        past_data, future_data = window.get_input_data()
        target_columns = window.target_columns
        past_dynamic_columns = window.past_dynamic_columns
        known_dynamic_columns = window.known_dynamic_columns

    past_data, future_data = _cast_fev_features(
        past_data=past_data,
        future_data=future_data,
        target_columns=target_columns,
        past_dynamic_columns=past_dynamic_columns,
        known_dynamic_columns=known_dynamic_columns,
    )

    num_series: int = len(past_data)
    num_past_covariates: int = len(past_dynamic_columns)
    num_future_covariates: int = len(known_dynamic_columns)

    # We use numpy format because torch does not support str covariates
    target_data = past_data.select_columns(target_columns).with_format("numpy")
    # past of past-only and known-future covariates
    dynamic_columns = [*past_dynamic_columns, *known_dynamic_columns]
    past_covariate_data = past_data.select_columns(dynamic_columns).with_format("numpy")
    future_known_data = future_data.select_columns(known_dynamic_columns).with_format("numpy")

    if num_past_covariates + num_future_covariates > 0:
        assert len(past_covariate_data) == num_series
    if num_future_covariates > 0:
        assert len(future_known_data) == num_series

    inputs: list[dict[str, torch.Tensor | dict[str, torch.Tensor]]] = []
    for idx, target_row in enumerate(target_data):
        target_row = cast(dict, target_row)
        # this assumes that the targets have the same length for multivariate tasks
        target_tensor_i = np.stack([target_row[col] for col in target_columns])
        entry: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {
            "target": torch.Tensor(rearrange(target_tensor_i,"c t -> t c"))
        }

        if len(dynamic_columns) > 0:
            past_covariate_row = past_covariate_data[idx]
            entry["past_covariates"] = {
                col: torch.Tensor(rearrange(past_covariate_row[col], "t -> t 1"))
                for col in dynamic_columns if past_covariate_row[col].dtype.kind in "fuib"
            }

        if len(known_dynamic_columns) > 0:
            future_known_row = future_known_data[idx]
            entry["future_covariates"] = {
                col: torch.Tensor(rearrange(future_known_row[col], "t -> t 1"))
                for col in known_dynamic_columns if future_known_row[col].dtype.kind in "fuib"
            }

        inputs.append(entry)

    return inputs, target_columns, past_dynamic_columns, known_dynamic_columns

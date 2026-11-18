"""Handle a missing target prefix on D-wabi's own Chronos FEV pipeline."""

from functools import wraps

import numpy as np


def _prepare_item(raw):
    target = np.asarray(raw["target"])
    if target.ndim not in (1, 2) or target.size != target.shape[-1] or np.isfinite(target).any():
        return raw
    # The pinned univariate target encoder needs at least one observed response.
    groups = {
        name: {
            key: np.full(value.shape, np.nan, dtype=np.float32)
            if isinstance(value, np.ndarray) and not np.issubdtype(value.dtype, np.number)
            else value
            for key, value in raw[name].items()
        }
        for name in ("past_covariates", "future_covariates")
        if name in raw
    }
    return raw | groups


def prepare_fev_inputs(predict_quantiles):
    """Wrap the per-instance call that receives Chronos' converted FEV rows."""

    @wraps(predict_quantiles)
    def predict(inputs, *args, **kwargs):
        return predict_quantiles([_prepare_item(raw) for raw in inputs], *args, **kwargs)

    return predict

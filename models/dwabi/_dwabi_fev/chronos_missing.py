"""Keep categorical covariates missing when Chronos has no target to encode them with."""

from functools import wraps

import numpy as np
from chronos.chronos2 import dataset


def install():
    original = dataset.validate_and_prepare_single_dict_input

    @wraps(original)
    def prepare(raw, *args, **kwargs):
        target = np.asarray(raw["target"])
        if target.ndim <= 2 and target.size == target.shape[-1] and not np.isfinite(target).any():
            groups = {}
            for name in ("past_covariates", "future_covariates"):
                if name in raw:
                    groups[name] = {
                        key: np.full(value.shape, np.nan, dtype=np.float32)
                        if isinstance(value, np.ndarray) and not np.issubdtype(value.dtype, np.number)
                        else value
                        for key, value in raw[name].items()
                    }
            raw = raw | groups
        return original(raw, *args, **kwargs)

    dataset.validate_and_prepare_single_dict_input = prepare

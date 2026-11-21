"""fev wrapper for ChakraTS-OS-10M, an open-weight forecasting model from YHat Labs
(https://huggingface.co/yhatlabs/ChakraTS-OS-10M), run locally with the `yhatmodels` library.

Each evaluation window's series go to the model with all of their target columns (forecast jointly), past covariates
and known covariates, and up to 32,768 steps of history, with the `pro` preset. With `cross_series=16`, single-target
series without covariates are read 16 at a time within the window, in order of history length. Non-numeric covariate
columns (text, booleans) are categories: target-encoded per series in single-target tasks, else ordinal codes over the
column's values. Static columns are not used.

The weights are gated on Hugging Face during early access (request access on the model page, then `hf auth login`) and
released under the Chakra Non-Commercial Model License. The first download needs a one-time license acceptance with a
yhatlabs.com account: run `yhatmodels accept-license ChakraTS-OS-10M` in a terminal, or set YHAT_MODELS_TOKEN.
"""

from __future__ import annotations

import datasets
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

import fev


def _numeric(table: pa.Table, col: str) -> bool:
    t = table.column(col).type.value_type
    return pa.types.is_floating(t) or pa.types.is_integer(t)


def _vocab(tables: list[pa.Table], col: str) -> pa.Array:
    """Sorted distinct non-null values of a non-numeric column across the context and future tables."""
    vals = [t.column(col).combine_chunks().values.cast(pa.string()) for t in tables]
    return pc.unique(pa.concat_arrays(vals).drop_null()).sort()


def _rows(table: pa.Table, col: str, vocab: pa.Array | None = None, text: bool = False) -> list[np.ndarray]:
    """One array per series of a list column: float32 numbers (nulls -> NaN), category text, or ordinal codes."""
    arr = table.column(col).combine_chunks()
    offsets = arr.offsets.to_numpy()
    vals = arr.values
    if vocab is None:
        flat = vals.cast(pa.float32()).to_numpy(zero_copy_only=False)
    elif text:
        flat = np.asarray(vals.cast(pa.string()).to_pylist(), dtype=object)
    else:
        flat = pc.index_in(vals.cast(pa.string()), value_set=vocab).cast(pa.float32()).to_numpy(zero_copy_only=False)
    return [flat[a:b] for a, b in zip(offsets[:-1], offsets[1:])]


class ChakraTSOSModel(fev.ForecastingModel):
    model_name = "chakrats-os"
    trained_on_datasets: list[str] = []

    def __init__(self, model: str = "yhatlabs/ChakraTS-OS-10M", preset: str = "pro", cross_series: int = 16, device: str | None = None):
        super().__init__()
        self.model_id, self.preset, self.cross_series, self.device = model, preset, cross_series, device
        self._model = None

    def _load(self):
        if self._model is None:  # loaded once, reused across tasks
            from yhatmodels import ChakraTS

            self._model = ChakraTS.from_pretrained(self.model_id, device=self.device)
        return self._model

    def _covariates(self, task: fev.Task, pt: pa.Table, ft: pa.Table | None, single: bool):
        """Per series: past covariates over the context, known covariates over the context and the horizon."""

        def column(tables: list[pa.Table], col: str) -> list[np.ndarray]:
            if _numeric(tables[0], col):
                parts = [_rows(t, col) for t in tables]
            else:
                vocab = _vocab(tables, col)
                parts = [_rows(t, col, vocab, text=single) for t in tables]
            return [np.concatenate(p) for p in zip(*parts)]

        past = {c: column([pt], c) for c in task.past_dynamic_columns}
        known = {c: column([pt, ft], c) for c in task.known_dynamic_columns}
        n = pt.num_rows
        past_list = [{c: v[i] for c, v in past.items()} for i in range(n)] if past else None
        known_list = [{c: v[i] for c, v in known.items()} for i in range(n)] if known else None
        return past_list, known_list

    def _fit_predict(self, task: fev.Task) -> list[datasets.DatasetDict | datasets.Dataset]:
        model = self._load()
        cols = list(task.target_columns)
        levels = sorted(set(task.quantile_levels) | {0.5})
        predictions = []
        for window in task.iter_windows():
            past, future = window.get_input_data()
            pt = past.data.table
            ft = future.data.table if future is not None else None
            per_target = [_rows(pt, c) for c in cols]
            targets = [np.stack(rows) for rows in zip(*per_target)]  # (V, T) per series
            past_cov, known_cov = self._covariates(task, pt, ft, single=len(cols) == 1)
            with self._record_inference_time():
                qs = model.forecast(targets, window.horizon, quantiles=levels, past=past_cov, known=known_cov,
                                    preset=self.preset, cross_series=self.cross_series)  # fmt: skip
            per_col = {}
            for j, c in enumerate(cols):
                per_col[c] = datasets.Dataset.from_dict(
                    {"predictions": [q[j, :, levels.index(0.5)].tolist() for q in qs],
                     **{str(lv): [q[j, :, levels.index(lv)].tolist() for q in qs] for lv in task.quantile_levels}}  # fmt: skip
                )
            predictions.append(datasets.DatasetDict(per_col) if len(cols) > 1 else per_col[cols[0]])
        return predictions

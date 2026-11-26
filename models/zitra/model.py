"""Official autogluon/fev ForecastingModel wrapper for Zitra v3 (100% Pure Apache-2.0)."""

from __future__ import annotations

from dataclasses import dataclass

import datasets
import fev
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc


@dataclass(frozen=True)
class ColumnKind:
    kind: str  # "numeric" | "bool" | "string"
    categories: tuple | None = None

    @property
    def categorical(self) -> bool:
        return self.kind != "numeric"


def infer_column_kind(values: np.ndarray) -> ColumnKind:
    series = pd.Series(values)
    non_null = series.dropna()
    values_are_bool = len(non_null) > 0 and bool(non_null.map(lambda v: isinstance(v, (bool, np.bool_))).all())
    if pd.api.types.is_bool_dtype(series) or values_are_bool:
        return ColumnKind("bool", (False, True))
    if pd.to_numeric(series, errors="coerce").notna().equals(series.notna()):
        return ColumnKind("numeric")
    return ColumnKind("string", tuple(pd.Categorical(series).categories))


def encode_values(values: np.ndarray, kind: ColumnKind) -> np.ndarray:
    if kind.kind == "string":
        codes = pd.Categorical(values, categories=list(kind.categories)).codes.astype(np.float64)
        codes[codes < 0] = np.nan
        return codes
    if values.dtype.kind == "f":
        return values.astype(np.float64)
    return pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=np.float64)


def _flat_column(table: pa.Table, col: str) -> tuple[np.ndarray, np.ndarray]:
    arr = table.column(col).combine_chunks()
    lengths = pc.list_value_length(arr).to_numpy(zero_copy_only=False).astype(np.int64)
    values = pc.list_flatten(arr).to_numpy(zero_copy_only=False)
    return values, np.concatenate([[0], np.cumsum(lengths)])


def _split(values: np.ndarray, offsets: np.ndarray) -> list[np.ndarray]:
    return [values[offsets[i] : offsets[i + 1]] for i in range(len(offsets) - 1)]


@dataclass
class Item:
    target: np.ndarray  # (V, T)
    past: np.ndarray  # (C_past, T)
    past_cat: np.ndarray  # (C_past_cat, T)
    known: np.ndarray  # (C_known, T + H)
    known_cat: np.ndarray  # (C_known_cat, T + H)

    @property
    def length(self) -> int:
        return self.target.shape[-1]

    @property
    def rows(self) -> int:
        return sum(a.shape[0] for a in (self.target, self.past, self.past_cat, self.known, self.known_cat))


class ZitraModel(fev.ForecastingModel):
    """Zitra v3 forecasting model (100% Apache-2.0, https://github.com/tomascupr/zitra-forecast).

    Supports Zitra-Titan-v3 via the unified `zitra.ZitraForecaster` API with multivariate targets
    plus past and known-future covariates (up to 16,384 context on known covariates).
    """

    model_name = "zitra"
    trained_on_datasets = []

    def __init__(
        self,
        model_path: str = "tomcupr/Zitra-Titan-v3",
        device: str = "auto",
        batch_budget: int = 2**21,
        max_batch_size: int = 64,
    ):
        super().__init__()
        self.model_path = model_path
        self.device = device
        self.batch_budget = batch_budget
        self.max_batch_size = max_batch_size
        self._forecaster = None

    def _load_forecaster(self):
        import torch
        from zitra import ZitraForecaster

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self._forecaster is None:
            self._forecaster = ZitraForecaster.from_pretrained(self.model_path, device=self.device)
        return self._forecaster

    def _window_items(self, task: fev.Task, past_data, future_data, kinds: dict[str, ColumnKind]) -> list[Item]:
        h = task.horizon
        cap = self._forecaster.max_context
        past_t, fut_t = past_data.data.table, future_data.data.table

        def rows(table, cols):
            out = []
            for c in cols:
                vals, offs = _flat_column(table, c)
                out.append(_split(encode_values(vals, kinds[c]) if c in kinds else vals.astype(np.float64), offs))
            return out

        targets = rows(past_t, task.target_columns)
        p_num = [c for c in task.past_dynamic_columns if not kinds[c].categorical]
        p_cat = [c for c in task.past_dynamic_columns if kinds[c].categorical]
        k_num = [c for c in task.known_dynamic_columns if not kinds[c].categorical]
        k_cat = [c for c in task.known_dynamic_columns if kinds[c].categorical]
        pn, pcat = rows(past_t, p_num), rows(past_t, p_cat)
        kn_p, kc_p = rows(past_t, k_num), rows(past_t, k_cat)
        kn_f, kc_f = rows(fut_t, k_num), rows(fut_t, k_cat)

        def stack(cols, length):
            return np.stack(cols) if cols else np.zeros((0, length))

        items = []
        for i in range(len(past_data)):
            t = len(targets[0][i])
            s = max(0, t - cap)
            items.append(
                Item(
                    target=stack([col[i][s:] for col in targets], t - s),
                    past=stack([col[i][s:] for col in pn], t - s),
                    past_cat=stack([col[i][s:] for col in pcat], t - s),
                    known=stack([np.concatenate([a[i][s:], b[i]]) for a, b in zip(kn_p, kn_f)], t - s + h),
                    known_cat=stack([np.concatenate([a[i][s:], b[i]]) for a, b in zip(kc_p, kc_f)], t - s + h),
                )
            )
        return items

    def _batches(self, items: list[Item], horizon: int) -> list[list[int]]:
        order = sorted(range(len(items)), key=lambda i: items[i].length)
        batches, cur, cur_max = [], [], 0
        for i in order:
            it = items[i]
            new_max = max(cur_max, it.length + horizon)
            rows = sum(items[j].rows for j in cur) + it.rows
            if cur and (rows * new_max > self.batch_budget or len(cur) >= self.max_batch_size):
                batches.append(cur)
                cur, new_max = [], it.length + horizon
            cur.append(i)
            cur_max = new_max
        if cur:
            batches.append(cur)
        return batches

    @staticmethod
    def _pad(arrays: list[np.ndarray], length: int) -> np.ndarray | None:
        if arrays[0].shape[0] == 0:
            return None
        out = np.full((len(arrays), arrays[0].shape[0], length), np.nan, dtype=np.float32)
        for b, a in enumerate(arrays):
            out[b, :, length - a.shape[-1] :] = a
        return out

    def _forecast_batch(self, batch: list[Item], horizon: int) -> np.ndarray:
        t = max(it.length for it in batch)
        f = self._forecaster.forecast(
            self._pad([it.target for it in batch], t),
            horizon,
            past_covariates=self._pad([it.past for it in batch], t),
            known_covariates=self._pad([it.known for it in batch], t + horizon),
            categorical_past_covariates=self._pad([it.past_cat for it in batch], t),
            categorical_known_covariates=self._pad([it.known_cat for it in batch], t + horizon),
        )
        return f.quantiles.double().cpu().numpy()

    def _forecast_items(self, items: list[Item], horizon: int) -> list[np.ndarray]:
        import torch

        out = [None] * len(items)
        pending = self._batches(items, horizon)
        while pending:
            idx = pending.pop(0)
            try:
                q = self._forecast_batch([items[i] for i in idx], horizon)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                if len(idx) == 1:
                    raise
                half = len(idx) // 2
                pending[:0] = [idx[:half], idx[half:]]
                continue
            for j, i in enumerate(idx):
                out[i] = q[j]
        return out

    def _to_levels(self, q: np.ndarray, levels: list[float]) -> np.ndarray:
        model_levels = list(self._forecaster.quantile_levels)
        cols = []
        for lv in levels:
            match = [k for k, m in enumerate(model_levels) if abs(m - lv) < 1e-9]
            if match:
                cols.append(q[..., match[0]])
            else:
                cols.append(np.apply_along_axis(lambda r: np.interp(lv, model_levels, r), -1, q))
        return np.stack(cols, axis=-1)

    def _fit_predict(self, task: fev.Task) -> list[datasets.DatasetDict]:
        forecaster = self._load_forecaster()
        table = task.load_full_dataset().data.table
        kinds = {col: infer_column_kind(_flat_column(table, col)[0]) for col in task.dynamic_columns}

        items, n_per_window = [], []
        for window in task.iter_windows():
            past_data, future_data = window.get_input_data()
            window_items = self._window_items(task, past_data, future_data, kinds)
            n_per_window.append(len(window_items))
            items += window_items

        with self._record_inference_time():
            quantiles = self._forecast_items(items, task.horizon)

        levels = list(task.quantile_levels)
        median_idx = list(forecaster.quantile_levels).index(0.5)
        h = task.horizon
        predictions, start = [], 0
        for n in n_per_window:
            q = np.stack(quantiles[start : start + n])
            start += n
            q_task = self._to_levels(q, levels)
            per_target = {}
            for v, col in enumerate(task.target_columns):
                columns = {"predictions": q[:, v, :, median_idx]}
                columns.update({str(lv): q_task[:, v, :, k] for k, lv in enumerate(levels)})
                per_target[col] = datasets.Dataset(
                    pa.table(
                        {
                            name: pa.FixedSizeListArray.from_arrays(pa.array(arr.reshape(-1), pa.float64()), h)
                            for name, arr in columns.items()
                        }
                    )
                )
            predictions.append(datasets.DatasetDict(per_target))
        return predictions

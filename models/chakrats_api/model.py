"""fev wrapper for ChakraTS, a hosted forecasting model from YHat Labs (https://yhatlabs.com).

ChakraTS is served through an API; no weights are distributed. Set YHAT_API_KEY (evaluation keys are
issued on request, see https://huggingface.co/yhatlabs/ChakraTS). The wrapper sends every item of a
window with all of its target columns, past covariates and known covariates, 100 items per call, and
never mixes evaluation windows in one call. The `chakra-ts-fev` model is the fev-bench evaluation
configuration of ChakraTS: it declares no overlap with fev-bench datasets.
"""

from __future__ import annotations

import json
import math
import os
import time
import urllib.error
import urllib.request

import datasets
import fev
import numpy as np
import pandas as pd
from fev.model import ForecastingModel

API_URL = os.environ.get("YHAT_API_URL", "https://api.yhatlabs.com/v1/forecast")
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
QKEYS = [str(q) for q in QUANTILES]
MAX_CONTEXT = 20480  # points of history sent per item; longer than every component's own context window


def _jsonable(v):
    """Numbers stay numbers (NaN -> null), text stays text; the API encodes text covariates itself."""
    if v is None:
        return None
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, str):
        return v
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return None if math.isnan(f) else f


def _freq(timestamps) -> str:
    idx = pd.DatetimeIndex(timestamps[:50])
    f = pd.infer_freq(idx) if len(idx) >= 3 else None
    if f is None and len(idx) >= 2:
        f = pd.tseries.frequencies.to_offset(idx[1] - idx[0]).freqstr
    return f or "h"


class ChakraTSAPIModel(ForecastingModel):
    model_name = "chakrats_api"
    trained_on_datasets: list[str] = []

    def __init__(self, model: str = "chakra-ts-fev", batch_size: int = 100, api_key: str | None = None):
        super().__init__()
        self.model = model
        self.batch_size = batch_size
        self.api_key = api_key or os.environ["YHAT_API_KEY"]

    def _call(self, items: list[dict], horizon: int, freq: str) -> list[dict]:
        body = {"model": self.model, "horizon": horizon, "freq": freq, "series": items}
        req = urllib.request.Request(
            API_URL,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}", "User-Agent": "chakrats-fev-wrapper/1.0 (+https://yhatlabs.com)"},
        )
        for attempt in range(8):
            try:
                with urllib.request.urlopen(req, timeout=3600) as r:
                    return json.load(r)["forecasts"]
            except urllib.error.HTTPError as e:
                if e.code in (429, 503) and attempt < 7:  # quota window, or the server waking up
                    time.sleep(min(300, 30 * 2**attempt))
                    continue
                raise RuntimeError(f"API error {e.code}: {e.read()[:300]!r}") from None
        raise RuntimeError("the API kept answering 429/503")

    def _fit_predict(self, task: fev.Task) -> list[datasets.DatasetDict]:
        cols = list(task.target_columns)
        known = list(task.known_dynamic_columns)
        past_cols = list(task.past_dynamic_columns)
        predictions = []
        for window in task.iter_windows():
            past, future = window.get_input_data()
            rows = list(past)
            fut = {str(r[task.id_column]): r for r in future} if future is not None else {}
            freq = _freq(rows[0][task.timestamp_column])
            items = []
            for r in rows:
                rid = str(r[task.id_column])
                item: dict = {"id": rid}
                if len(cols) == 1:
                    item["values"] = [_jsonable(v) for v in list(r[cols[0]])[-MAX_CONTEXT:]]
                else:
                    item["targets"] = {c: [_jsonable(v) for v in list(r[c])[-MAX_CONTEXT:]] for c in cols}
                if past_cols:
                    item["past_covariates"] = {c: [_jsonable(v) for v in list(r[c])[-MAX_CONTEXT:]] for c in past_cols}
                if known:
                    item["known_covariates"] = {
                        c: [_jsonable(v) for v in list(r[c])[-MAX_CONTEXT:]] + [_jsonable(v) for v in fut[rid][c]] for c in known
                    }
                items.append(item)
            out = {}
            with self._record_inference_time():
                for i in range(0, len(items), self.batch_size):
                    for f in self._call(items[i : i + self.batch_size], window.horizon, freq):
                        out[f["id"]] = f
            gt_ids = [str(r[task.id_column]) for r in window.get_ground_truth()]
            q_of = lambda f, c: np.asarray(f["quantiles"] if c is None else f["targets"][c], dtype=np.float32)  # [h, 9]
            per_col = {}
            for c in cols:
                arrs = [q_of(out[g], None if len(cols) == 1 else c) for g in gt_ids]
                per_col[c] = datasets.Dataset.from_dict(
                    {"predictions": [a[:, 4].tolist() for a in arrs], **{k: [a[:, j].tolist() for a in arrs] for j, k in enumerate(QKEYS)}}
                )
            predictions.append(datasets.DatasetDict(per_col) if len(cols) > 1 else per_col[cols[0]])
        return predictions

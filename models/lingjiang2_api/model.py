"""FEV adapter for the hosted LingJiang 2.0 forecasting API."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import lzma
import os
import time
import uuid
from types import SimpleNamespace

import datasets
import fev
import numpy as np


QUANTILES = tuple(str(level / 10) for level in range(1, 10))
MAX_CONTEXT = 15360
MAX_RESPONSE_BYTES = 256 * 1024 * 1024
MAX_AICS_REQUEST_BYTES = 900_000
MAX_CHUNK_CHARS = 820_000


def _numeric(values):
    array = np.asarray(values)
    if array.dtype.kind in "biuf":
        return array.astype(np.float32)
    try:
        return np.asarray(
            [np.nan if value is None or value == "" else float(value) for value in values],
            dtype=np.float32,
        )
    except (TypeError, ValueError):
        pass
    encoded = []
    for value in values:
        if value is None or value == "":
            encoded.append(np.nan)
        else:
            digest = hashlib.sha256(str(value).encode("utf-8")).digest()
            encoded.append(float(int.from_bytes(digest[:4], "little") % 1000000))
    return np.asarray(encoded, dtype=np.float32)


def _make_inputs(task, window):
    past, future = window.get_input_data()
    if len(past) != len(future):
        raise ValueError("Past and future item counts differ")
    items = []
    for old, new in zip(past, future, strict=True):
        if str(old[task.id_column]) != str(new[task.id_column]):
            raise ValueError("Past and future item IDs differ")
        context = np.stack([_numeric(old[column]) for column in task.target_columns])
        past_length = context.shape[-1]
        context = context[:, -MAX_CONTEXT:]
        history = context.shape[-1]
        past_covariates = None
        if task.past_dynamic_columns:
            past_covariates = np.stack(
                [_numeric(old[column])[-history:] for column in task.past_dynamic_columns]
            )
        known_covariates = None
        if task.known_dynamic_columns:
            rows = []
            for column in task.known_dynamic_columns:
                past_values = _numeric(old[column])
                future_values = _numeric(new[column])
                if len(past_values) != past_length or len(future_values) != task.horizon:
                    raise ValueError(f"Invalid length for known covariate {column}")
                rows.append(np.concatenate([past_values[-history:], future_values]))
            known_covariates = np.stack(rows)
        items.append(SimpleNamespace(
            context=context,
            past_covariates=past_covariates,
            known_covariates=known_covariates,
        ))
    return items


def _predictions_dataset(task, forecasts):
    if not forecasts:
        raise ValueError("Empty API prediction")
    targets = {}
    for target_index, target in enumerate(task.target_columns):
        columns = {"predictions": []}
        columns.update({quantile: [] for quantile in QUANTILES})
        for forecast in forecasts:
            array = np.asarray(forecast, dtype=np.float32)
            if array.shape != (9, task.horizon, len(task.target_columns)):
                raise ValueError("Invalid API prediction shape")
            if not np.isfinite(array).all():
                raise ValueError("API prediction contains non-finite values")
            columns["predictions"].append(array[4, :, target_index].astype(float).tolist())
            for quantile_index, quantile in enumerate(QUANTILES):
                columns[quantile].append(
                    array[quantile_index, :, target_index].astype(float).tolist()
                )
        targets[target] = datasets.Dataset.from_dict(
            columns, features=task.predictions_schema
        )
    return datasets.DatasetDict(targets)


def _pack_array(value):
    if value is None:
        return None
    stream = io.BytesIO()
    np.save(stream, np.asarray(value, dtype=np.float32), allow_pickle=False)
    return base64.b64encode(stream.getvalue()).decode("ascii")


def _unpack_array(value):
    if not isinstance(value, str):
        raise ValueError("Expected an encoded prediction array")
    array = np.load(io.BytesIO(base64.b64decode(value, validate=True)), allow_pickle=False)
    if array.dtype != np.float32 or array.size == 0:
        raise ValueError("Invalid prediction array")
    return array


def _encode_request(batch, horizon):
    payload = {
        "schema_version": 1,
        "prediction_length": horizon,
        "contexts": [_pack_array(item.context) for item in batch],
        "past_only_covariates": [_pack_array(item.past_covariates) for item in batch],
        "past_future_covariates": [_pack_array(item.known_covariates) for item in batch],
    }
    return base64.b64encode(
        lzma.compress(json.dumps(payload, allow_nan=False).encode("utf-8"))
    ).decode("ascii")


def _request_params(batch, horizon):
    params = {"input": _encode_request(batch, horizon)}
    wire_bytes = len(json.dumps(params, separators=(",", ":")).encode("utf-8"))
    return params, wire_bytes


def _pad_batch(batch, compute_size):
    if not batch or not len(batch) <= compute_size <= 32:
        raise ValueError("Invalid compute batch size")
    return list(batch) + [batch[0]] * (compute_size - len(batch))


def _cloud_plans(items, max_items, horizon):
    if max_items < 1:
        raise ValueError("max_items must be positive")

    def split(batch, compute_size):
        _, wire_bytes = _request_params(_pad_batch(batch, compute_size), horizon)
        if wire_bytes <= MAX_AICS_REQUEST_BYTES or len(batch) == 1:
            yield batch, compute_size
        else:
            middle = len(batch) // 2
            yield from split(batch[:middle], compute_size)
            yield from split(batch[middle:], compute_size)

    for start in range(0, len(items), max_items):
        original = items[start:start + max_items]
        yield from split(original, len(original))


def _decode_response(value):
    for _ in range(8):
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                raw = lzma.LZMADecompressor()
                data = raw.decompress(
                    base64.b64decode(value, validate=True), max_length=MAX_RESPONSE_BYTES + 1
                )
                if len(data) > MAX_RESPONSE_BYTES or not raw.eof:
                    raise ValueError("Incomplete or oversized API response")
                value = json.loads(data)
        if isinstance(value, dict):
            if value.get("schema_version") == 1:
                return value
            if value.get("success") is False:
                raise RuntimeError("AICS invocation failed")
            for key in ("data", "output", "result", "re", "value"):
                if key in value:
                    value = value[key]
                    break
            else:
                raise ValueError("Unrecognized API response")
        else:
            raise ValueError("Unrecognized API response type")
    raise ValueError("API response nesting is too deep")


class _AICSClient:
    def __init__(self, service_id, timeout=3600):
        from alibabacloud_brain_industrial20200920.client import Client
        from alibabacloud_brain_industrial20200920.models import AicsOpenApiInvokeRequest
        from alibabacloud_tea_openapi.models import Config

        config = Config(
            access_key_id=os.environ["ALIBABA_CLOUD_ACCESS_KEY_ID"],
            access_key_secret=os.environ["ALIBABA_CLOUD_ACCESS_KEY_SECRET"],
            region_id="cn-hangzhou",
            read_timeout=60000,
            connect_timeout=60000,
        )
        config.endpoint = "brain-industrial.cn-hangzhou.aliyuncs.com"
        self.client = Client(config)
        self.request_type = AicsOpenApiInvokeRequest
        self.service_id = service_id
        self.timeout = timeout

    def _invoke(self, params):
        job_id = uuid.uuid4().hex
        deadline = time.monotonic() + self.timeout
        errors = 0
        while time.monotonic() < deadline:
            request = self.request_type(
                service_id=self.service_id, param=params,
                type="EXPERIMENT", job_id=job_id,
            )
            try:
                response = self.client.aics_open_api_invoke(request)
            except Exception as exc:
                errors += 1
                retryable = str(getattr(exc, "status_code", "")) in (
                    "429", "500", "502", "503", "504"
                )
                if errors >= (12 if retryable else 3):
                    raise
                if retryable:
                    params = {"input": ""}
                time.sleep(min(30, 3 * 2 ** min(errors - 1, 3)))
                continue
            data = response.body.data
            errors = 0
            if isinstance(data, str):
                data = json.loads(data)
            status = data.get("jobStatus")
            if status in ("FAIL", "FAILED", "FAILURE", "CANCELLED", "CANCELED", "ERROR", "STOPPED"):
                raise RuntimeError(f"AICS job ended with status {status}")
            if status in (None, "SUCCESS"):
                for key in ("output", "result"):
                    if data.get(key) not in (None, "null", ""):
                        return _decode_response(data[key])
            params = {"input": ""}
            time.sleep(3)
        raise TimeoutError(f"AICS job exceeded {self.timeout} seconds")

    def predict(self, batch, horizon, target_count, compute_size=None):
        padded = _pad_batch(batch, compute_size or len(batch))
        params, wire_bytes = _request_params(padded, horizon)
        if wire_bytes <= MAX_AICS_REQUEST_BYTES:
            payload = self._invoke(params)
        else:
            encoded = params["input"]
            transfer_id = uuid.uuid4().hex
            chunk_chars = MAX_CHUNK_CHARS
            while True:
                parts = [encoded[start:start + chunk_chars]
                         for start in range(0, len(encoded), chunk_chars)]
                if len(parts) > 128:
                    raise ValueError("Chunked request exceeds 128 API calls")
                requests = []
                for index, part in enumerate(parts):
                    chunk = {"schema_version": 1, "transport": "chunk-v1",
                             "transfer_id": transfer_id, "chunk_index": index,
                             "chunk_count": len(parts), "data": part}
                    request = {"input": base64.b64encode(
                        lzma.compress(json.dumps(chunk, allow_nan=False).encode("utf-8"))
                    ).decode("ascii")}
                    size = len(json.dumps(request, separators=(",", ":")).encode())
                    requests.append((request, size))
                if all(size <= MAX_AICS_REQUEST_BYTES for _, size in requests):
                    break
                chunk_chars = int(chunk_chars * 0.8)
                if chunk_chars < 50_000:
                    raise ValueError("API chunk cannot fit the request limit")
            for index, (request, _) in enumerate(requests):
                result = self._invoke(request)
                if index < len(parts) - 1:
                    if (result.get("transport") != "chunk-ack-v1" or
                            result.get("transfer_id") != transfer_id or
                            result.get("chunk_index") != index):
                        raise ValueError("Invalid chunk acknowledgment")
                else:
                    payload = result
        if payload.get("schema_version") != 1 or payload.get("model") != "lingjiang2-only":
            raise ValueError("Unexpected API model or schema")
        if payload.get("forecast_keys") != list(QUANTILES):
            raise ValueError("Unexpected forecast quantiles")
        forecasts = [_unpack_array(item) for item in payload["forecasts"]]
        if len(forecasts) != len(padded):
            raise ValueError("Unexpected forecast count")
        for forecast in forecasts:
            if forecast.shape != (9, horizon, target_count) or not np.isfinite(forecast).all():
                raise ValueError("Unexpected forecast shape or value")
        return forecasts[:len(batch)]


class LingJiang2API(fev.ForecastingModel):
    """Zero-shot forecasts through the hosted LingJiang 2.0 API."""

    model_name = "lingjiang2_api"
    trained_on_datasets = []

    def __init__(self, service_id=None, batch_uni=32, batch_multi=8):
        super().__init__()
        self.service_id = service_id or os.environ["LINGJIANG2_FEV_CLOUD_SERVICE_ID"]
        self.batch_uni = batch_uni
        self.batch_multi = batch_multi
        self.client = None

    def _fit_predict(self, task: fev.Task):
        if self.client is None:
            self.client = _AICSClient(self.service_id)
        predictions = []
        for window in task.iter_windows():
            inputs = _make_inputs(task, window)
            batch_size = self.batch_multi if len(task.target_columns) > 1 else self.batch_uni
            forecasts = []
            for batch, compute_size in _cloud_plans(inputs, batch_size, task.horizon):
                with self._record_inference_time():
                    forecasts.extend(
                        self.client.predict(batch, task.horizon, len(task.target_columns),
                                            compute_size=compute_size)
                    )
            predictions.append(_predictions_dataset(task, forecasts))
        return predictions

"""Store and reuse verified recipient-aligned, outcome-free earlier forecasts."""

import json
from hashlib import sha256
from importlib.metadata import version
from inspect import getsource
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

FORMAT = "dwabi-fev-forecast-bank-v2"


def json_bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()


def array_hash(values):
    values = np.asarray(values)
    if values.dtype.hasobject:
        raise TypeError("Forecast-bank array hashes require a concrete dtype.")
    values = np.ascontiguousarray(values.astype(values.dtype.newbyteorder("<"), copy=False))
    digest = sha256(json_bytes([values.dtype.str, values.shape]))
    digest.update(values)
    return digest.hexdigest()


def column_hash(column):
    """Hash logical typed values, not Arrow padding, slice tails or chunk layout."""
    validity, sizes, values = sha256(), sha256(), sha256()
    children = []
    dtype = column.type
    for original in column.chunks:
        # Bound temporary allocations even when the source has one large chunk.
        for start in range(0, len(original), 65_536):
            chunk = original.slice(start, 65_536)
            validity.update(chunk.is_valid().to_numpy(zero_copy_only=False).astype("u1"))
            if pa.types.is_list(dtype) or pa.types.is_large_list(dtype) or pa.types.is_fixed_size_list(dtype):
                lengths = pc.fill_null(pc.list_value_length(chunk), 0).to_numpy(zero_copy_only=False)
                sizes.update(np.asarray(lengths, dtype="<i8"))
                children.append(pc.list_flatten(chunk))
            elif (
                pa.types.is_string(dtype)
                or pa.types.is_large_string(dtype)
                or pa.types.is_binary(dtype)
                or pa.types.is_large_binary(dtype)
            ):
                if chunk.null_count:
                    chunk = pc.fill_null(
                        chunk, b"" if pa.types.is_binary(dtype) or pa.types.is_large_binary(dtype) else ""
                    )
                _, offsets_buffer, data = chunk.buffers()
                width = 8 if pa.types.is_large_string(dtype) or pa.types.is_large_binary(dtype) else 4
                offsets = np.frombuffer(
                    offsets_buffer, dtype=f"<i{width}", count=len(chunk) + 1, offset=chunk.offset * width
                )
                sizes.update(np.asarray(np.diff(offsets), dtype="<i8"))
                if data is not None:
                    values.update(memoryview(data)[int(offsets[0]) : int(offsets[-1])])
            elif pa.types.is_null(dtype):
                continue
            elif (
                pa.types.is_integer(dtype)
                or pa.types.is_floating(dtype)
                or pa.types.is_boolean(dtype)
                or pa.types.is_temporal(dtype)
            ):
                if pa.types.is_temporal(dtype):
                    if not (
                        pa.types.is_date(dtype)
                        or pa.types.is_time(dtype)
                        or pa.types.is_timestamp(dtype)
                        or pa.types.is_duration(dtype)
                    ):
                        raise TypeError(f"Forecast bank does not support the Arrow temporal type {dtype}.")
                    chunk = chunk.view(pa.int32() if dtype.bit_width == 32 else pa.int64())
                if chunk.null_count:
                    chunk = pc.fill_null(chunk, pa.scalar(False if pa.types.is_boolean(dtype) else 0, type=chunk.type))
                numeric = chunk.to_numpy(zero_copy_only=False)
                if numeric.dtype.kind == "f" and np.isnan(numeric).any():
                    numeric = numeric.copy()
                    numeric[np.isnan(numeric)] = np.nan
                values.update(np.ascontiguousarray(numeric.astype(numeric.dtype.newbyteorder("<"), copy=False)))
            else:
                raise TypeError(f"Forecast bank does not support the Arrow input type {dtype}.")
    child = column_hash(pa.chunked_array(children, type=dtype.value_type)) if children else None
    return sha256(
        json_bytes([str(dtype), len(column), validity.hexdigest(), sizes.hexdigest(), values.hexdigest(), child])
    ).hexdigest()


def dataset_hash(dataset, fields):
    if set(dataset.column_names) != set(fields):
        raise ValueError("Forecast-bank inputs must contain only the declared issue-time fields.")
    digest = sha256(
        json_bytes(
            {"features": dataset.features.to_dict(), "format": dataset.format, "fields": fields, "rows": len(dataset)}
        )
    )
    # Dataset.iter respects logical selection indices. Formatting this view does
    # not change the provider's dataset, and never touches window.full_dataset.
    for batch in dataset.select_columns(fields).with_format("arrow").iter(batch_size=32):
        for name in fields:
            digest.update(bytes.fromhex(column_hash(batch.column(name))))
    return digest.hexdigest()


def provider_inputs(task, case, ends):
    window, past, future, ids, dates, _ = case
    common = [task.id_column, task.timestamp_column, *task.known_dynamic_columns, *task.static_columns]
    fields = [*common, *task.target_columns, *task.past_dynamic_columns]
    settings = (
        "cutoff",
        "horizon",
        "min_context_length",
        "max_context_length",
        "id_column",
        "timestamp_column",
        "target_columns",
        "known_dynamic_columns",
        "past_dynamic_columns",
        "static_columns",
    )
    return {
        "past": dataset_hash(past, fields),
        "future": dataset_hash(future, common),
        "ids": sha256(json_bytes(list(ids))).hexdigest(),
        "dates": array_hash(dates),
        "ends": array_hash(ends),
        "task_definition": task.to_dict(),
        "frequency": task.freq,
        "window": {name: getattr(window, name) for name in settings},
    }


def provider_binding(model, name):
    """The JSON input pins alone do not bind hard-coded adapter options."""
    root = Path(__file__).resolve().parent
    files = [root / "providers.py", root / "cases.py"]
    if name in ("t0", "ci", "timestral"):
        files.extend(root / leaf for leaf in ("members.py", "types.py"))
    extra = {
        "ci": "_ci_runtime.py",
        "timestral": "_timestral_runtime.py",
        "chronos2": "chronos_missing.py",
        "toto2": "toto_short_history.py",
    }
    if name in extra:
        files.append(root / extra[name])
    pin = model.provider_pins[name]
    if "wrapper" in pin:
        files.append(root.parents[1] / pin["wrapper"] / "model.py")
    return {
        "pin": pin,
        "dispatch": sha256(getsource(model._forecast).encode()).hexdigest(),
        "sources": {
            path.relative_to(root.parents[1]).as_posix(): sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for path in files
        },
    }


class ForecastBank:
    def __init__(self, root, inputs):
        self.root = Path(root)
        self.inputs = inputs

    def forecast(self, model, task, row, case, ends, members):
        laws = {}
        observed = provider_inputs(task, case, ends)
        for name in members:
            path = self.root / name / task.task_name / f"{row['origin']}.npz"
            expected = {
                "format": FORMAT,
                "provider_inputs": observed,
                "provider_binding": provider_binding(model, name),
                "inputs": self.inputs,
                "member": name,
                "task": task.task_name,
                "issue": row,
                "cutoffs_sha256": array_hash(ends.astype("<i8")),
                "lead_times_sha256": array_hash(case[4].astype("<i8")),
                "known": list(task.known_dynamic_columns),
                "past_only": list(task.past_dynamic_columns),
                "static": list(task.static_columns),
                "deciles": list(task.quantile_levels),
            }
            expected = json.loads(json_bytes(expected))
            if not path.exists():
                law = model._forecast(task, case, ends, (name,))[name]
                metadata = expected | {
                    "law_sha256": array_hash(law),
                    "producer": {"device": model.device, "torch": version("torch"), "dwabi": version("dwabi")},
                }
                path.parent.mkdir(parents=True, exist_ok=True)
                temp = None
                try:
                    with NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
                        temp = Path(stream.name)
                        np.savez_compressed(stream, law=law, metadata=json.dumps(metadata, allow_nan=False))
                    try:
                        path.hardlink_to(temp)
                    except FileExistsError:
                        pass  # Reuse and verify the first complete writer's receipt.
                finally:
                    if temp is not None:
                        temp.unlink()
            try:
                with np.load(path, allow_pickle=False) as saved:
                    metadata = json.loads(str(saved["metadata"].item()))
                    producer = metadata.pop("producer")
                    law_hash = metadata.pop("law_sha256")
                    if (
                        metadata != expected
                        or producer["dwabi"] != version("dwabi")
                        or not producer["device"]
                        or not producer["torch"]
                    ):
                        raise ValueError("Incomplete or changed provider-input binding.")
                    law = saved["law"]
                if law_hash != array_hash(law):
                    raise ValueError("The cached law changed.")
            except (KeyError, TypeError, ValueError, OSError) as error:
                raise ValueError(
                    f"{path}: incompatible forecast cache; use a fresh bank directory. "
                    "Existing receipts are never upgraded or overwritten."
                ) from error
            shape = (*case[4].shape, len(task.target_columns), len(task.quantile_levels) + 1)
            if law.shape != shape or not np.isfinite(law).all():
                raise ValueError(f"{path}: the forecast does not cover the complete native recipient axes.")
            laws[name] = law
        return laws

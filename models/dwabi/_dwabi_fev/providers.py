"""Load the pinned zero-shot reference providers."""

from hashlib import file_digest, sha256
from importlib.metadata import version
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from threading import RLock, get_ident
from types import ModuleType, SimpleNamespace
import sys

import numpy as np
import pyarrow.compute as pc
from fev import ForecastingModel

from . import _ci_runtime
from .cases import typed_panel
from .members import CIProtoMember, T0Member, TimestralMember


_WRAPPER_LOCK = RLock()
_TIREX_LOCK = RLock()


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def checkpoint(pin, root, member):
    if version(pin["package"]) != pin["version"]:
        raise ValueError(f"{member}: the runtime version differs from the benchmark pin.")
    if root is None:
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(pin["repo"], revision=pin["revision"], allow_patterns=list(pin["weights"])))
    else:
        path = Path(root) / member
    if any(digest(path / name) != expected for name, expected in pin["weights"].items()):
        raise ValueError(f"{member}: checkpoint differs from the benchmark pin.")
    return path


def _wrapper_digest(path):
    return sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _registered_namespace(model, pin, member):
    function = model.__dict__.get("_fit_predict")
    namespace = getattr(function, "__globals__", {})
    try:
        source = Path(namespace["__file__"])
        verified = (namespace.get("__name__") == model.__module__
                    and namespace.get(model.__name__) is model
                    and Path(function.__code__.co_filename).resolve() == source.resolve()
                    and _wrapper_digest(source) == pin["wrapper_sha256"])
    except (KeyError, OSError, TypeError, AttributeError, ValueError):
        verified = False
    if not verified:
        raise ValueError(f"{member}: the registered upstream adapter differs from the pinned source.")
    return namespace


def wrapper(pin, member):
    path = Path(__file__).resolve().parents[2] / pin["wrapper"] / "model.py"
    if _wrapper_digest(path) != pin["wrapper_sha256"]:
        raise ValueError(f"{member}: the upstream adapter differs from the benchmark pin.")
    name = f"_dwabi_provider_{member}"
    with _WRAPPER_LOCK:
        registered = ForecastingModel._registry.get(pin["wrapper"])
        namespace = None if registered is None else _registered_namespace(registered, pin, member)
        module = sys.modules.get(name)
        if module is not None:
            try:
                verified_source = _wrapper_digest(module.__file__) == pin["wrapper_sha256"]
            except (OSError, TypeError, AttributeError, ValueError):
                verified_source = False
            if (registered is None or module.__dict__.get(registered.__name__) is not registered
                    or not verified_source):
                raise ValueError(f"{member}: the loaded upstream adapter differs from the pinned source.")
            return module
        if registered is not None:
            module = sys.modules.get(registered.__module__)
            if module is None:
                # FEV's file loader can register a class without retaining its module.
                module = ModuleType(registered.__module__)
                module.__dict__.update(namespace)
            elif module.__dict__ is not namespace:
                raise ValueError(f"{member}: the registered module differs from the pinned source.")
            sys.modules[name] = module
            return module
        spec = spec_from_file_location(name, path)
        module = module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
        return module


def native_array(task, predictions, n):
    predictions = task.clean_and_validate_predictions(predictions)
    columns = ("predictions", *map(str, task.quantile_levels))
    targets = []
    for target in task.target_columns:
        table = predictions[target].data.table
        if len(table) != n:
            raise ValueError("The adapter changed the number of forecast recipients.")
        values = [pc.list_flatten(table[name]).to_numpy(zero_copy_only=False) for name in columns]
        targets.append(np.stack(values, axis=-1).reshape(n, task.horizon, len(columns)))
    return np.stack(targets, axis=2).astype(float)


def make_provider(member, pin, root, device="cuda"):
    if member == "timesfm3" and sys.flags.hash_randomization:
        raise ValueError("TimesFM3 requires starting Python with PYTHONHASHSEED=0. "
                         "Set it before launching the interpreter; changing os.environ after startup is insufficient.")
    import torch

    torch.manual_seed(1937)
    torch.set_num_threads(4)
    path = checkpoint(pin, root, member)
    if member in ("t0", "ci", "timestral"):
        if member == "ci":
            model = CIProtoMember(path, device, frequency=None)
            rotary = _ci_runtime.prepare_fixed_rotary(device, model.model.backend)
        else:
            model = {"t0": T0Member, "timestral": TimestralMember}[member](path, device)
            rotary = False

        def adapter(task, case):
            if member == "ci":
                model.frequency = task.freq
            with _ci_runtime.rotary_scope(rotary):
                law = model(typed_panel(task, case[1], case[2])).values
            return np.concatenate((law[..., 4:5], law), axis=-1)

        return adapter
    if member == "tirex":
        from tirex2 import load_model
        from tirex2.model.component import flashrnn_slstm

        with _TIREX_LOCK:
            original_backend = flashrnn_slstm._flashrnn_backend
            constructing_thread = get_ident()

            def backend(runtime_device):
                return "vanilla" if get_ident() == constructing_thread else original_backend(runtime_device)

            flashrnn_slstm._flashrnn_backend = backend
            try:
                model = load_model(path, device=device)
            finally:
                flashrnn_slstm._flashrnn_backend = original_backend

        def adapter(task, case):
            law = model.forecast_fev(case[0], prediction_length=task.horizon, output_type="fev",
                                     batch_size=64, quantile_levels=task.quantile_levels)
            return native_array(task, law, len(case[3]))

        return adapter
    if member == "chronos2":
        from chronos import BaseChronosPipeline

        from .chronos_missing import prepare_fev_inputs

        model = BaseChronosPipeline.from_pretrained(str(path), device_map=device, torch_dtype=torch.float32)
        model.predict_quantiles = prepare_fev_inputs(model.predict_quantiles)

        def adapter(task, case):
            view = SimpleNamespace(iter_windows=lambda: iter((case[0],)), quantile_levels=task.quantile_levels)
            laws, _ = model.predict_fev(view, batch_size=100, cross_learning=True, as_univariate=False)
            return native_array(task, laws[0], len(case[3]))

        return adapter
    module = wrapper(pin, member)
    if member == "timeh2o":
        from h2ofm import TimeH2O

        model = module.TimeH2OModel(device=device, batch_size=64)
        model._model = TimeH2O.from_pretrained(path, device=device, batch_size=64,
                                               local_files_only=True, token=False)

        def adapter(task, case):
            view = SimpleNamespace(iter_windows=lambda: iter((case[0],)), quantile_levels=task.quantile_levels)
            law, = model._fit_predict(view)
            if list(law) != task.target_columns:
                raise ValueError("TimeH2O changed the native target order.")
            columns = ("predictions", *map(str, task.quantile_levels))
            arrays = []
            for target in task.target_columns:
                data = law[target].with_format("numpy")
                arrays.append(np.stack([np.asarray(data[name]) for name in columns], axis=-1))
            values = np.stack(arrays, axis=2)
            if (not np.array_equal(values[..., 0], values[..., 5])
                    or (np.diff(values[..., 1:], axis=-1) < 0.).any()):
                raise ValueError("TimeH2O must return ordered deciles and their median as its point.")
            return values

        return adapter
    if member == "toto2":
        from .toto_short_history import toto_short_history

        model = module.Toto2Model(model_path=str(path), device=device, batch_size=16)

        def adapter(task, case):
            view = SimpleNamespace(iter_windows=lambda: iter((case[0],)), target_columns=task.target_columns,
                                   quantile_levels=task.quantile_levels, horizon=task.horizon)
            with toto_short_history(task, case):
                law, = model._fit_predict(view)
            return native_array(task, law, len(case[3]))

        return adapter
    if member != "timesfm3":
        raise ValueError(f"Unknown benchmark member {member}.")
    model = module.TimesFM3Model(checkpoint_path=str(path), device=device, min_batch=1, max_batch=8,
                                 per_core_batch_size=2)
    model._get_forecaster()

    def adapter(task, case):
        view = SimpleNamespace(iter_windows=lambda: iter((case[0],)), target_columns=task.target_columns,
                               known_dynamic_columns=task.known_dynamic_columns,
                               past_dynamic_columns=task.past_dynamic_columns, quantile_levels=task.quantile_levels)
        law, = model._fit_predict(view)
        # Preserve the stock wrapper's raw prediction precision before scoring.
        columns = ("predictions", *map(str, task.quantile_levels))
        values = [[pc.list_flatten(law[target].data.table[name]).to_numpy(zero_copy_only=False)
                   for name in columns] for target in task.target_columns]
        return np.asarray(values, dtype=np.float64).reshape(
            len(values), len(columns), len(case[3]), task.horizon
        ).transpose(2, 3, 0, 1)

    return adapter

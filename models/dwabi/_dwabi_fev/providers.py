"""Load the pinned zero-shot reference providers."""

from hashlib import file_digest, sha256
from importlib.metadata import version
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.compute as pc

from . import _ci_runtime
from .cases import typed_panel
from .members import CIProtoMember, T0Member, TimestralMember


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


def wrapper(pin, member):
    path = Path(__file__).resolve().parents[2] / pin["wrapper"] / "model.py"
    if sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() != pin["wrapper_sha256"]:
        raise ValueError(f"{member}: the upstream adapter differs from the benchmark pin.")
    spec = spec_from_file_location(f"_dwabi_provider_{member}", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
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

        flashrnn_slstm._flashrnn_backend = lambda device: "vanilla"
        model = load_model(path, device=device)

        def adapter(task, case):
            law = model.forecast_fev(case[0], prediction_length=task.horizon, output_type="fev",
                                     batch_size=64, quantile_levels=task.quantile_levels)
            return native_array(task, law, len(case[3]))

        return adapter
    if member == "chronos2":
        from chronos import BaseChronosPipeline

        from .chronos_missing import install

        install()
        model = BaseChronosPipeline.from_pretrained(str(path), device_map=device, torch_dtype=torch.float32)

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

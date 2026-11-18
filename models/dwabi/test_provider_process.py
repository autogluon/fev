"""CPU regressions for process-wide upstream adapters and hash-seed startup."""

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from _dwabi_fev import providers

from fev import ForecastingModel

ROOT = Path(__file__).resolve().parent
PINS = json.loads((ROOT / "provider_pins.json").read_text())


@pytest.fixture
def isolated_registry(monkeypatch):
    original = ForecastingModel._registry.copy()
    for member in ("toto2", "timesfm3", "timeh2o"):
        ForecastingModel._registry.pop(PINS[member]["wrapper"], None)
        monkeypatch.delitem(sys.modules, f"_dwabi_provider_{member}", raising=False)
    yield
    ForecastingModel._registry.clear()
    ForecastingModel._registry.update(original)
    for member in ("toto2", "timesfm3", "timeh2o"):
        monkeypatch.delitem(sys.modules, f"_dwabi_provider_{member}", raising=False)


@pytest.mark.parametrize("member, attribute", [
    ("toto2", "Toto2Model"), ("timesfm3", "TimesFM3Model"), ("timeh2o", "TimeH2OModel"),
])
def test_repeated_wrapper_requests_reuse_the_verified_module(isolated_registry, member, attribute):
    first = providers.wrapper(PINS[member], member)
    second = providers.wrapper(PINS[member], member)
    assert first is second
    assert getattr(first, attribute) is ForecastingModel.get_model_cls(PINS[member]["wrapper"])


@pytest.mark.parametrize("register_module", [True, False])
def test_prior_upstream_import_reuses_its_verified_class(isolated_registry, monkeypatch, register_module):
    pin = PINS["toto2"]
    path = ROOT.parent / pin["wrapper"] / "model.py"
    name = "_test_already_imported_toto"
    spec = spec_from_file_location(name, path)
    original = module_from_spec(spec)
    if register_module:
        monkeypatch.setitem(sys.modules, name, original)
    spec.loader.exec_module(original)
    reused = providers.wrapper(pin, "toto2")
    assert reused.Toto2Model is original.Toto2Model
    assert providers.wrapper(pin, "toto2") is reused
    if register_module:
        assert reused is original


def test_mismatched_registered_class_is_rejected(isolated_registry):
    class UnverifiedToto(ForecastingModel):
        model_name = "toto-2.0"

        def _fit_predict(self, task):
            return []

    with pytest.raises(ValueError, match="pinned source"):
        providers.wrapper(PINS["toto2"], "toto2")
    assert ForecastingModel.get_model_cls("toto-2.0") is UnverifiedToto


def test_occupied_loader_module_is_not_silently_trusted(isolated_registry, monkeypatch):
    unverified = ModuleType("_dwabi_provider_toto2")
    unverified.__file__ = __file__
    monkeypatch.setitem(sys.modules, unverified.__name__, unverified)
    with pytest.raises(ValueError, match="pinned source"):
        providers.wrapper(PINS["toto2"], "toto2")
    assert sys.modules[unverified.__name__] is unverified


def test_cached_wrapper_still_checks_the_requested_pin(isolated_registry):
    providers.wrapper(PINS["toto2"], "toto2")
    wrong = PINS["toto2"] | {"wrapper_sha256": "0"*64}
    with pytest.raises(ValueError, match="benchmark pin"):
        providers.wrapper(wrong, "toto2")


@pytest.mark.parametrize("fail", [False, True])
def test_tirex_backend_is_constructor_local_and_restored(monkeypatch, fail):
    import tirex2
    from tirex2.model.component import flashrnn_slstm

    original = flashrnn_slstm._flashrnn_backend
    monkeypatch.setattr(providers, "checkpoint", lambda *args: Path("unused"))

    def load_model(path, device):
        assert flashrnn_slstm._flashrnn_backend(device) == "vanilla"
        with ThreadPoolExecutor(max_workers=1) as workers:
            assert workers.submit(flashrnn_slstm._flashrnn_backend, device).result(timeout=5) == original(device)
        if fail:
            raise RuntimeError("injected construction failure")
        return SimpleNamespace()

    monkeypatch.setattr(tirex2, "load_model", load_model)
    if fail:
        with pytest.raises(RuntimeError, match="injected construction failure"):
            providers.make_provider("tirex", {}, None, device="cuda")
    else:
        providers.make_provider("tirex", {}, None, device="cuda")
    assert flashrnn_slstm._flashrnn_backend is original


def test_chronos_compatibility_wraps_only_the_created_pipeline(monkeypatch):
    from chronos import BaseChronosPipeline
    from chronos.chronos2 import dataset

    original_parser = dataset.validate_and_prepare_single_dict_input
    original_predict = lambda *args, **kwargs: None
    wrapped_predict = lambda *args, **kwargs: None
    pipeline = SimpleNamespace(predict_quantiles=original_predict)
    calls = []

    def prepare(call):
        calls.append(call)
        return wrapped_predict

    monkeypatch.setattr(providers, "checkpoint", lambda *args: Path("unused"))
    monkeypatch.setattr(BaseChronosPipeline, "from_pretrained", lambda *args, **kwargs: pipeline)
    monkeypatch.setitem(sys.modules, "_dwabi_fev.chronos_missing", SimpleNamespace(prepare_fev_inputs=prepare))
    providers.make_provider("chronos2", {}, None, device="cpu")
    assert calls == [original_predict]
    assert pipeline.predict_quantiles is wrapped_predict
    assert dataset.validate_and_prepare_single_dict_input is original_parser


@pytest.mark.parametrize("seed, late, options, allowed", [
    ("0", False, (), True),
    ("1", False, (), False),
    (None, False, (), False),
    ("random", False, (), False),
    ("1", True, (), False),
    (None, True, (), False),
    ("0", False, ("-R",), False),
])
def test_timesfm_seed_is_checked_before_checkpoint_preparation(seed, late, options, allowed):
    script = """
import os
import sys
if sys.argv[1] == "late":
    os.environ["PYTHONHASHSEED"] = "0"
from _dwabi_fev import providers

class CheckpointReached(Exception):
    pass

def checkpoint(*args):
    raise CheckpointReached

providers.checkpoint = checkpoint
try:
    providers.make_provider("timesfm3", {}, None, device="cpu")
except CheckpointReached:
    print("checkpoint reached")
except ValueError as error:
    assert "PYTHONHASHSEED=0" in str(error), str(error)
    print("startup seed rejected")
"""
    environment = os.environ.copy()
    environment.update(PYTHONPATH=os.pathsep.join((str(ROOT.parents[1] / "src"), str(ROOT))),
                       CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", NUMBA_NUM_THREADS="1")
    if seed is None:
        environment.pop("PYTHONHASHSEED", None)
    else:
        environment["PYTHONHASHSEED"] = seed
    result = subprocess.run([sys.executable, *options, "-B", "-c", script, "late" if late else "startup"],
                            env=environment, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ("checkpoint reached" if allowed else "startup seed rejected")

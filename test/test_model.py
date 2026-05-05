import importlib.util
import time
from pathlib import Path

import pytest

from fev.model import ForecastingModel

MODELS_DIR = Path(__file__).parent.parent / "models"


class DummyModel(ForecastingModel):
    def _fit_predict(self, task):
        with self._record_training_time():
            time.sleep(0.01)
        with self._record_inference_time():
            time.sleep(0.01)
        return []


class AnotherFancyModel(ForecastingModel):
    def _fit_predict(self, task):
        return []


def test_when_concrete_subclass_defined_then_it_is_registered():
    assert "dummy" in ForecastingModel.list_available_models()
    assert "anotherfancy" in ForecastingModel.list_available_models()


def test_when_get_model_cls_called_then_correct_class_returned():
    assert ForecastingModel.get_model_cls("dummy") is DummyModel
    assert ForecastingModel.get_model_cls("Dummy") is DummyModel
    assert DummyModel.model_name == "dummy"


def test_when_unknown_model_requested_then_error_is_raised():
    with pytest.raises(ValueError, match="Unknown model"):
        ForecastingModel.get_model_cls("nonexistent")


def test_when_abstract_class_instantiated_then_error_is_raised():
    with pytest.raises(TypeError):
        ForecastingModel()


def test_when_fit_predict_called_then_timing_is_reset_and_recorded():
    model = DummyModel()
    model.training_time = 99.0
    model.inference_time = 99.0
    model.fit_predict(task=None)
    assert 0.01 <= model.training_time < 1.0
    assert 0.01 <= model.inference_time < 1.0


def test_when_fit_predict_called_twice_then_timing_reflects_only_last_call():
    model = DummyModel()
    model.fit_predict(task=None)
    model.fit_predict(task=None)
    assert model.training_time < 1.0
    assert model.inference_time < 1.0


def test_when_model_name_set_then_registered_under_custom_name():
    class MyCustomModel(ForecastingModel):
        model_name = "my-custom-model-v3"

        def _fit_predict(self, task):
            return []

    assert "my-custom-model-v3" in ForecastingModel.list_available_models()
    assert ForecastingModel.get_model_cls("my-custom-model-v3") is MyCustomModel


def test_when_duplicate_model_name_registered_then_error_is_raised():
    with pytest.raises(ValueError, match="already registered"):

        class DummyModel(ForecastingModel):  # noqa: F811
            def _fit_predict(self, task):
                return []


def test_when_model_wrapper_imported_then_folder_name_matches_model_name():
    """Ensure each model folder registers exactly one model_name equal to the folder name."""
    import re

    model_folders = [p.parent for p in MODELS_DIR.glob("*/model.py")]
    assert len(model_folders) > 0, "No model folders found"
    for folder in model_folders:
        model_file = folder / "model.py"
        content = model_file.read_text()
        matches = re.findall(r'model_name\s*=\s*"([^"]+)"', content)
        assert matches == [folder.name], (
            f'Expected exactly model_name = "{folder.name}" in {model_file}, got: {matches}'
        )

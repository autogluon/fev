import pytest

from fev.model import ForecastingModel


class DummyModel(ForecastingModel):
    def fit_predict(self, task):
        return []


class AnotherFancyModel(ForecastingModel):
    def fit_predict(self, task):
        return []


def test_when_concrete_subclass_defined_then_it_is_registered():
    assert "dummy" in ForecastingModel.list_available_models()
    assert "anotherfancy" in ForecastingModel.list_available_models()


def test_when_get_model_cls_called_then_correct_class_returned():
    assert ForecastingModel.get_model_cls("dummy") is DummyModel
    assert ForecastingModel.get_model_cls("Dummy") is DummyModel


def test_when_unknown_model_requested_then_error_is_raised():
    with pytest.raises(ValueError, match="Unknown model"):
        ForecastingModel.get_model_cls("nonexistent")


def test_when_abstract_class_instantiated_then_error_is_raised():
    with pytest.raises(TypeError):
        ForecastingModel()


def test_when_record_training_time_used_then_training_time_updated():
    model = DummyModel()
    assert model.training_time == 0.0
    with model._record_training_time():
        pass
    assert model.training_time > 0.0


def test_when_record_inference_time_used_then_inference_time_updated():
    model = DummyModel()
    assert model.inference_time == 0.0
    with model._record_inference_time():
        pass
    assert model.inference_time > 0.0

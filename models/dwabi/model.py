"""Adapt the packaged D-wabi forecaster to FEV."""

from pathlib import Path

from dwabi.fev import PRETRAINED_ON, NativeFEV
from fev.model import ForecastingModel


class DWabiFEVModel(ForecastingModel):
    model_name = "dwabi"
    trained_on_datasets = list(PRETRAINED_ON)

    def __init__(self, **kwargs):
        super().__init__()
        self.runtime = NativeFEV(fev_root=Path(__file__).resolve().parents[2], **kwargs)

    def _fit_predict(self, task):
        with self._record_training_time():
            fitted = self.runtime.fit(task)
        with self._record_inference_time():
            return self.runtime.predict(task, fitted)

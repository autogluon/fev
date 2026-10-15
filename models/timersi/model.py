"""TimeRSI's full search and chronological composition, from its standalone package."""

import os

from timersi_u1.fev_adapter import TimeRSIConfig
from timersi_u1.task_process import fit_predict

from fev.model import ForecastingModel


class TimeRSIModel(ForecastingModel):
    model_name = "timersi"
    trained_on_datasets = []

    def __init__(
        self,
        checkpoint_path="google/timesfm-3.0-pytorch",
        vertex_project=None,
        run_root="timersi-runs",
    ):
        super().__init__()
        self.config = TimeRSIConfig(
            checkpoint_path=checkpoint_path,
            vertex_project=vertex_project or os.environ.get("TIMERSI_VERTEX_PROJECT"),
            run_root=run_root,
        )

    def _fit_predict(self, task):
        forecasts, timing, _ = fit_predict(task, self.config)
        self.training_time = timing["training_time_s"]
        self.inference_time = timing["inference_time_s"]
        return forecasts

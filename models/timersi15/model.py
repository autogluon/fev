"""FEV interface for the standalone TimeRSI-1.5 implementation."""
from fev.model import ForecastingModel
from timersi15.runtime import Config, fit_predict


class TimeRSI15Model(ForecastingModel):
    model_name = "timersi15"
    trained_on_datasets = []

    def __init__(self, checkpoint_path="mistralai/Timestral-209M", device="cuda:0",
                 run_root="timersi15-runs", regenerate_decisions=False,
                 agent_model="claude-opus-5", effort="high", vertex_project=None, artifacts_dir=None):
        super().__init__()
        self.config = Config(checkpoint_path=checkpoint_path, device=device,
            run_root=run_root, regenerate_decisions=regenerate_decisions,
            agent_model=agent_model, effort=effort, vertex_project=vertex_project, artifacts_dir=artifacts_dir)

    def _fit_predict(self, task):
        forecasts, timing = fit_predict(task, self.config)
        self.training_time = timing["training_time_s"]
        self.inference_time = timing["inference_time_s"]
        return forecasts

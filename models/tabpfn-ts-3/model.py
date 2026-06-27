import datasets

import fev


class TabPFNTS3Model(fev.ForecastingModel):
    """TabPFN-TS-3 model from https://github.com/PriorLabs/tabpfn-time-series."""

    model_name = "tabpfn-ts-3"

    def __init__(
        self,
        max_context_length: int = 32768,
        model_path: str = "tabpfn-v3-regressor-v3_20260506_timeseries.ckpt",
        use_covariates: bool = True,
    ):
        super().__init__()
        self.max_context_length = max_context_length
        self.model_path = model_path
        self.use_covariates = use_covariates

    def _fit_predict(self, task: fev.Task) -> list[datasets.DatasetDict]:
        from tabpfn_time_series import TabPFNMode, TabPFNTSPipeline

        pipeline = TabPFNTSPipeline(
            tabpfn_mode=TabPFNMode.LOCAL,
            tabpfn_model_config={"model_path": fev.utils.maybe_cache_from_s3(self.model_path)},
        )
        predictions_per_window, self.inference_time = pipeline.predict_fev(task, use_covariates=self.use_covariates)
        return predictions_per_window

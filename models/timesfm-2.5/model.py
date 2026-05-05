import datasets
import numpy as np

import fev


class TimesFM25Model(fev.ForecastingModel):
    """TimesFM 2.5 model from https://github.com/google-research/timesfm."""

    model_name = "timesfm-2.5"
    trained_on_datasets = [
        "favorita_transactions_1D",
        "favorita_transactions_1W",
        "fred_md_2025",
        "proenfo_gfc12",
        "proenfo_gfc14",
        "proenfo_gfc17",
        "kdd_cup_2022_10T",
        "m5_1D",
        "m5_1W",
    ]
    TIMESFM_MODEL_QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    def __init__(
        self,
        model_path: str = "google/timesfm-2.5-200m-pytorch",
        batch_size: int = 256,
        context_length: int = 16_000,
        per_core_batch_size: int = 64,
        device: str = "auto",
    ):
        super().__init__()
        self.model_path = model_path
        self.batch_size = batch_size
        self.context_length = context_length
        self.per_core_batch_size = per_core_batch_size
        self.device = device

    def _fit_predict(self, task: fev.Task) -> list[datasets.DatasetDict]:
        import timesfm
        import torch

        torch.set_float32_matmul_precision("high")

        if self.device == "auto":
            self.device = "gpu" if torch.cuda.is_available() else "cpu"

        context_length = min(
            self.context_length, max([len(t) for t in task.load_full_dataset()[task.timestamp_column]])
        )
        print(f"Setting context_length={context_length}")

        model_path = fev.utils.maybe_cache_from_s3(self.model_path)
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(model_path)
        model.compile(
            timesfm.ForecastConfig(
                max_context=context_length,
                max_horizon=task.horizon,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
                per_core_batch_size=self.per_core_batch_size,
            )
        )

        predictions_per_window = []
        for window in task.iter_windows():
            predictions = self._predict_window(
                window,
                model=model,
                quantile_levels=task.quantile_levels,
                return_mean=task.eval_metric in ["MSE", "RMSE", "RMSSE"],
            )
            predictions_per_window.append(predictions)
        return predictions_per_window

    def _predict_window(
        self,
        window: fev.EvaluationWindow,
        model,
        quantile_levels: list[float],
        return_mean: bool,
    ) -> datasets.DatasetDict:
        from gluonts.transform import LastValueImputation

        quantile_to_index = {}
        task_quantiles = [0.5] + quantile_levels
        for q in task_quantiles:
            quantile_to_index[q] = int(np.argmin(np.abs(np.array(self.TIMESFM_MODEL_QUANTILES) - q))) + 1

        past_data, _ = fev.convert_input_data(window, adapter="datasets", as_univariate=True)
        past_data = past_data.with_format("numpy").cast_column("target", datasets.Sequence(datasets.Value("float32")))

        imputation = LastValueImputation()
        inputs = [imputation(t.copy()) for t in past_data["target"]]

        forecast_batches = []
        with self._record_inference_time():
            for batch in self._batchify(inputs, batch_size=self.batch_size):
                mean_forecast, full_forecast = model.forecast(inputs=batch, horizon=window.horizon)
                if return_mean:
                    forecast = {"predictions": mean_forecast}
                else:
                    forecast = {"predictions": full_forecast[:, :, quantile_to_index[0.5]]}
                for q in quantile_levels:
                    forecast[str(q)] = full_forecast[:, :, quantile_to_index[q]]
                forecast_batches.append(forecast)

        predictions = datasets.Dataset.from_dict(
            {
                k: np.concatenate([batch[k] for batch in forecast_batches], axis=0)
                for k in ["predictions"] + [str(q) for q in quantile_levels]
            }
        )
        return fev.utils.combine_univariate_predictions_to_multivariate(
            predictions, target_columns=window.target_columns
        )

    @staticmethod
    def _batchify(lst: list, batch_size: int):
        for i in range(0, len(lst), batch_size):
            yield lst[i : i + batch_size]

from tempfile import TemporaryDirectory

import datasets

import fev


class AutoGluonModel(fev.ForecastingModel):
    """AutoGluon TimeSeriesPredictor from autogluon.timeseries."""

    model_name = "autogluon"

    def __init__(
        self,
        init_kwargs: dict | None = None,
        fit_kwargs: dict | None = None,
        fit_every_window: bool = False,
    ):
        super().__init__()
        self.init_kwargs = init_kwargs or {}
        self.fit_kwargs = fit_kwargs or {}
        self.fit_every_window = fit_every_window

    @staticmethod
    def _get_ag_eval_metric(fev_eval_metric: str) -> str:
        return "WQL" if fev_eval_metric == "MQL" else fev_eval_metric

    def _fit_predict(self, task: fev.Task) -> list[datasets.DatasetDict]:
        from autogluon.timeseries import TimeSeriesPredictor

        predictor = None
        predictions_per_window = []

        for idx, window in enumerate(task.iter_windows()):
            train_data, known_covariates = fev.convert_input_data(window, adapter="autogluon", as_univariate=True)

            if idx == 0 or self.fit_every_window:
                predictor = TimeSeriesPredictor(
                    prediction_length=task.horizon,
                    quantile_levels=task.quantile_levels,
                    eval_metric=self._get_ag_eval_metric(task.eval_metric),
                    known_covariates_names=list(known_covariates.columns),
                    path=TemporaryDirectory().name,
                    **self.init_kwargs,
                )
                with self._record_training_time():
                    predictor.fit(train_data, **self.fit_kwargs)

            with self._record_inference_time():
                forecast_df = predictor.predict(train_data, known_covariates=known_covariates)

            predictions_per_window.append(
                fev.utils.convert_forecast_df_to_predictions(
                    forecast_df.rename(columns={"mean": "predictions"}).to_data_frame().reset_index(),
                    horizon=window.horizon,
                    quantile_levels=task.quantile_levels,
                    target_columns=window.target_columns,
                )
            )

        return predictions_per_window

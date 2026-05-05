import warnings

import datasets
import pandas as pd

import fev


class StatsForecastModel(fev.ForecastingModel):
    """Model wrapper for the StatsForecast library."""

    model_name = "statsforecast"

    def __init__(
        self,
        model: str = "seasonalnaive",
        max_context_length: int | None = 1_000,
        max_season_length: int | None = None,
    ):
        super().__init__()
        self.model = model.lower().replace("-", "").replace("_", "").replace(" ", "")
        self.max_context_length = max_context_length
        self.max_season_length = max_season_length

    def _get_models(self, seasonality: int) -> list:
        from statsforecast.models import (
            AutoARIMA,
            AutoCES,
            AutoETS,
            AutoTheta,
            DynamicOptimizedTheta,
            Naive,
            RandomWalkWithDrift,
            SeasonalNaive,
        )

        models = {
            "naive": [Naive()],
            "drift": [RandomWalkWithDrift()],
            "seasonalnaive": [SeasonalNaive(season_length=seasonality)],
            "autoarima": [AutoARIMA(season_length=seasonality)],
            "autoets": [AutoETS(season_length=seasonality)],
            "autotheta": [AutoTheta(season_length=seasonality)],
            "statensemble": [
                AutoETS(season_length=seasonality),
                AutoARIMA(season_length=seasonality),
                DynamicOptimizedTheta(season_length=seasonality),
                AutoCES(season_length=seasonality),
            ],
        }
        if self.model not in models:
            raise ValueError(f"Unknown model '{self.model}'. Available: {list(models)}")
        return models[self.model]

    def _fit_predict(self, task: fev.Task) -> list[datasets.DatasetDict]:
        from autogluon.timeseries.utils.datetime import get_seasonality
        from statsforecast import StatsForecast
        from statsforecast.models import SeasonalNaive

        task.load_full_dataset()
        seasonality = max(task.seasonality, get_seasonality(task.freq))
        if self.max_season_length is not None and seasonality > self.max_season_length:
            seasonality = 1
        if self.max_context_length is not None and seasonality > self.max_context_length:
            seasonality = 1
        models = self._get_models(seasonality=seasonality)
        predictor = StatsForecast(
            freq="D",
            models=models,
            fallback_model=SeasonalNaive(season_length=seasonality),
            n_jobs=-1,
            verbose=False,
        )

        predictions_per_window = []
        for window in task.iter_windows():
            predictions = self._predict_window(window, predictor, quantile_levels=task.quantile_levels)
            predictions_per_window.append(predictions)
        return predictions_per_window

    def _predict_window(
        self, window: fev.EvaluationWindow, predictor, quantile_levels: list[float]
    ) -> datasets.DatasetDict:
        train_df, _, _ = fev.convert_input_data(window, adapter="nixtla", as_univariate=True)

        if train_df["y"].isna().any():
            train_df["y"] = train_df.groupby("unique_id", sort=False)["y"].ffill()
            train_df = train_df.fillna(0.0)

        if (train_df["ds"] > pd.Timestamp.max).any():
            train_df["ds"] = train_df.groupby("unique_id", sort=False).cumcount()

        if self.max_context_length is not None:
            train_df = (
                train_df.groupby("unique_id", as_index=False).tail(self.max_context_length).reset_index(drop=True)
            )

        levels = sorted(set([round(abs(q - 0.5) * 200) for q in quantile_levels]))
        with self._record_inference_time():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                forecast_df = predictor.forecast(h=window.horizon, df=train_df[["unique_id", "ds", "y"]], level=levels)

        model_names = [str(m) for m in predictor.models]
        forecast_df["predictions"] = forecast_df[model_names].median(1)
        for q in quantile_levels:
            suffix = self._quantile_to_suffix(q)
            forecast_df[str(q)] = forecast_df[[m + suffix for m in model_names]].median(1).to_numpy()

        forecast_df = forecast_df.fillna(0.0)

        return fev.utils.convert_forecast_df_to_predictions(
            forecast_df,
            horizon=window.horizon,
            quantile_levels=quantile_levels,
            target_columns=window.target_columns,
        )

    @staticmethod
    def _quantile_to_suffix(q: float) -> str:
        if q < 0.5:
            return f"-lo-{int(100 - 200 * q)}"
        return f"-hi-{int(200 * q - 100)}"

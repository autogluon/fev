import os
import warnings
from contextlib import contextmanager

import datasets
import pandas as pd

import fev


@contextmanager
def _disable_tqdm():
    old_value = os.environ.get("TQDM_DISABLE")
    os.environ["TQDM_DISABLE"] = "1"
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop("TQDM_DISABLE", None)
        else:
            os.environ["TQDM_DISABLE"] = old_value


class TabPFNTSModel(fev.ForecastingModel):
    """TabPFN-TS model from https://github.com/PriorLabs/tabpfn-time-series."""

    model_name = "tabpfn-ts"

    def __init__(self, max_context_length: int = 5000):
        super().__init__()
        self.max_context_length = max_context_length

    def _fit_predict(self, task: fev.Task) -> list[datasets.DatasetDict]:
        from tabpfn_time_series import FeatureTransformer, TabPFNMode, TabPFNTimeSeriesPredictor, TimeSeriesDataFrame
        from tabpfn_time_series.features import AutoSeasonalFeature, CalendarFeature, RunningIndexFeature

        predictor = TabPFNTimeSeriesPredictor(tabpfn_mode=TabPFNMode.LOCAL)
        selected_features = [RunningIndexFeature(), CalendarFeature(), AutoSeasonalFeature()]
        feature_transformer = FeatureTransformer(selected_features)

        predictions_per_window = []
        selected_columns = ["predictions"] + [str(q) for q in task.quantile_levels]
        for window in task.iter_windows():
            train_tsdf, test_tsdf, _ = fev.convert_input_data(window, "pandas", as_univariate=True)
            for df in [train_tsdf, test_tsdf]:
                for col in df.columns:
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = df[col].astype(str).replace("nan", "None")
            train_tsdf = TimeSeriesDataFrame(train_tsdf, id_column="id").fill_missing_values().fillna(0.0)
            train_tsdf = train_tsdf.slice_by_timestep(-self.max_context_length, None)
            train_tsdf = train_tsdf.drop(columns=task.past_dynamic_columns)
            test_tsdf = TimeSeriesDataFrame(test_tsdf, id_column="id").fill_missing_values().fillna(0.0)
            test_tsdf = test_tsdf.assign(target=float("nan"))
            train_tsdf, test_tsdf = feature_transformer.transform(train_tsdf, test_tsdf)
            with _disable_tqdm():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with self._record_inference_time():
                        forecast_df = predictor.predict(train_tsdf, test_tsdf)
            forecast_df = forecast_df.rename(
                columns={"target": "predictions"} | {q: str(q) for q in task.quantile_levels}
            )[selected_columns]
            predictions_per_window.append(
                fev.utils.convert_forecast_df_to_predictions(
                    forecast_df,
                    horizon=window.horizon,
                    quantile_levels=task.quantile_levels,
                    target_columns=window.target_columns,
                )
            )
        return predictions_per_window

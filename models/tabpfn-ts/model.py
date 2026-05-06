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


def _handle_missing_values(tsdf):
    """Fill NaN targets with 0 for series with <=1 valid values, drop NaN rows for others."""
    valid_counts = tsdf["target"].notna().groupby(level="item_id").sum()
    invalid_items = valid_counts[valid_counts <= 1].index

    result = tsdf.copy()

    if len(invalid_items) > 0:
        mask_to_fill = result.index.get_level_values("item_id").isin(invalid_items) & result["target"].isna()
        result.loc[mask_to_fill, "target"] = 0

    result = result[result["target"].notna()]
    return result


class TabPFNTSModel(fev.ForecastingModel):
    """TabPFN-TS model from https://github.com/PriorLabs/tabpfn-time-series."""

    model_name = "tabpfn-ts"

    def __init__(
        self,
        max_context_length: int = 4096,
        model_path: str = "tabpfn-v2-regressor-2noar4o2.ckpt",
        use_covariates: bool = True,
    ):
        super().__init__()
        self.max_context_length = max_context_length
        self.model_path = model_path
        self.use_covariates = use_covariates

    def _fit_predict(self, task: fev.Task) -> list[datasets.DatasetDict]:
        from tabpfn_time_series import FeatureTransformer, TabPFNMode, TabPFNTimeSeriesPredictor, TimeSeriesDataFrame
        from tabpfn_time_series.features import AutoSeasonalFeature, CalendarFeature, RunningIndexFeature

        predictor = TabPFNTimeSeriesPredictor(
            tabpfn_mode=TabPFNMode.LOCAL, tabpfn_config={"model_path": self.model_path}
        )
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
            train_tsdf = TimeSeriesDataFrame(train_tsdf, id_column="id")
            train_tsdf = _handle_missing_values(train_tsdf)
            train_tsdf = train_tsdf.slice_by_timestep(-self.max_context_length, None)
            test_tsdf = TimeSeriesDataFrame(test_tsdf, id_column="id")
            if not self.use_covariates and task.known_dynamic_columns:
                train_tsdf = train_tsdf.drop(columns=task.known_dynamic_columns)
                test_tsdf = test_tsdf.drop(columns=task.known_dynamic_columns)
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

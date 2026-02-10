"""MLForecast recursive models with LightGBM and CatBoost.

Design notes:
- Lightweight HPO tunes preprocessing only (differencing, scaling, lag transforms)
- Lags and time features selected via heuristics based on frequency
- No HPO of model hyperparameters to keep runtime manageable

This is the simplest setup we found that produces reasonable results across all 100 tasks
of fev-bench. Removing HPO or the heuristics significantly hurts performance. There's
likely room for better automated logic - contributions welcome!
"""

import time
import warnings
from typing import Literal

import datasets
import pandas as pd
from tqdm.auto import tqdm

import fev

datasets.disable_progress_bars()


def _create_lgbm(fit_time_limit: float | None, **model_kwargs):
    from lightgbm import LGBMRegressor
    from lightgbm.callback import EarlyStopException

    class _LGBMRegressor(LGBMRegressor):
        def fit(self, X, y, **kwargs):
            if fit_time_limit is not None:
                start_time = time.time()

                def _time_callback(env):
                    if time.time() - start_time >= fit_time_limit:
                        raise EarlyStopException(env.iteration, [])

                _time_callback.order = 30
                callbacks = kwargs.get("callbacks", []) or []
                kwargs["callbacks"] = callbacks + [_time_callback]
            return super().fit(X, y, **kwargs)

    return _LGBMRegressor(objective="mae", verbose=-1, **model_kwargs)


def _create_catboost(fit_time_limit: float | None, **model_kwargs):
    from catboost import CatBoostRegressor

    class _CatBoostRegressor(CatBoostRegressor):
        def fit(self, X, y=None, **kwargs):
            if "cat_features" not in kwargs:
                cat_cols = list(X.select_dtypes(include=["object", "category"]).columns)
                if cat_cols:
                    kwargs["cat_features"] = cat_cols

            if fit_time_limit is not None:
                start_time = time.time()

                class _TimeCallback:
                    def __init__(self):
                        self.time_end = start_time + fit_time_limit

                    def after_iteration(self, info):
                        time_cur = time.time()
                        time_per_iter = (time_cur - start_time) / max(info.iteration, 1)
                        return self.time_end >= (time_cur + 2 * time_per_iter)

                callbacks = kwargs.get("callbacks", []) or []
                kwargs["callbacks"] = callbacks + [_TimeCallback()]

            return super().fit(X, y, **kwargs)

    return _CatBoostRegressor(loss_function="MAE", verbose=False, allow_writing_files=False, **model_kwargs)


class MLForecastModel:
    """MLForecast with LightGBM or CatBoost regressor."""

    def __init__(
        self,
        regressor: Literal["lightgbm", "catboost"] = "lightgbm",
        lags: list[int] | None = None,
        date_features: list | None = None,
        differences: list[int] | None = None,
        fit_time_limit: float | None = 600,
        model_kwargs: dict | None = None,
    ):
        self.regressor = regressor
        self.lags = lags
        self.date_features = date_features
        self.differences = differences
        self.fit_time_limit = fit_time_limit
        self.model_kwargs = model_kwargs or {}

    def _create_model(self):
        if self.regressor == "lightgbm":
            return _create_lgbm(self.fit_time_limit, **self.model_kwargs)
        if self.regressor == "catboost":
            return _create_catboost(self.fit_time_limit, **self.model_kwargs)
        raise ValueError(f"Unknown regressor: {self.regressor}")

    def _get_lags(self, freq: str, median_series_len: int, seasonality: int = 1) -> list[int]:
        if self.lags is not None:
            return [lag for lag in self.lags if lag < median_series_len]

        from autogluon.timeseries.utils.datetime import get_lags_for_frequency

        diff_cost = max(self.differences) if self.differences else seasonality
        effective_len = median_series_len - diff_cost
        max_lag = min(effective_len - 1, max(1, effective_len - 10))
        lags = [lag for lag in get_lags_for_frequency(freq) if lag <= max_lag]

        if effective_len < 30 and len(lags) > 5:
            lags = lags[:5]

        return lags if lags else [1]

    def _get_date_features(self, freq: str) -> list:
        if self.date_features is not None:
            return self.date_features
        from autogluon.timeseries.utils.datetime import get_time_features_for_frequency

        return get_time_features_for_frequency(freq)

    def _get_target_transforms(self, seasonality: int = 1, min_series_len: int | None = None):
        from mlforecast.target_transforms import Differences, LocalStandardScaler

        transforms = []
        differences = self.differences if self.differences is not None else [seasonality]
        if differences and (min_series_len is None or min_series_len > max(differences)):
            transforms.append(Differences(differences))
        transforms.append(LocalStandardScaler())
        return transforms

    def _prepare_data(self, window: fev.EvaluationWindow, task: fev.Task) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        train_df, future_df, static_df = fev.convert_input_data(window, adapter="nixtla", as_univariate=True)
        train_df = train_df.copy()
        future_df = future_df.copy()

        if train_df["y"].isna().any():
            train_df["y"] = train_df.groupby("unique_id", sort=False)["y"].ffill().fillna(0.0)

        if task.past_dynamic_columns:
            train_df = train_df.drop(columns=task.past_dynamic_columns)

        if static_df is not None and len(static_df.columns) > 1:
            train_df = train_df.merge(static_df, on="unique_id", how="left")
            future_df = future_df.merge(static_df, on="unique_id", how="left")

        cat_cols = list(train_df.select_dtypes(include=["object", "category"]).columns)
        for col in cat_cols:
            train_df[col] = train_df[col].fillna("_NA_")
            future_df[col] = future_df[col].fillna("_NA_")
            all_categories = pd.concat([train_df[col], future_df[col]]).unique()
            cat_type = pd.CategoricalDtype(categories=all_categories)
            train_df[col] = train_df[col].astype(cat_type)
            future_df[col] = future_df[col].astype(cat_type)

        for df in [train_df, future_df]:
            num_cols = list(df.select_dtypes(include=["number"]).columns)
            df[num_cols] = df[num_cols].astype("float32")

        if len(future_df.columns) <= 2:
            future_df = None

        return train_df, future_df

    def _create_mlforecast(self, freq: str, lags: list[int], date_features: list, target_transforms: list):
        from mlforecast import MLForecast

        return MLForecast(
            models={self.regressor: self._create_model()},
            freq=freq,
            lags=lags,
            date_features=date_features,
            target_transforms=target_transforms,
        )

    def _format_predictions(
        self, preds_df: pd.DataFrame, quantile_levels: list[float], target_columns: list[str]
    ) -> datasets.DatasetDict:
        preds_df[fev.constants.PREDICTIONS] = preds_df[self.regressor]
        for q in quantile_levels:
            preds_df[str(q)] = preds_df[self.regressor]

        output_columns = [fev.constants.PREDICTIONS] + [str(q) for q in quantile_levels]
        predictions = [group[output_columns].to_dict("list") for _, group in preds_df.groupby("unique_id", sort=True)]
        return fev.utils.combine_univariate_predictions_to_multivariate(predictions, target_columns=target_columns)

    def _predict_window(
        self, window: fev.EvaluationWindow, task: fev.Task, tuned_config: dict | None = None
    ) -> tuple[datasets.DatasetDict, float, float]:
        train_df, future_df = self._prepare_data(window, task)

        if tuned_config is not None:
            lags = tuned_config["lags"]
            date_features = tuned_config["date_features"]
            target_transforms = tuned_config["target_transforms"]
        else:
            series_lengths = train_df.groupby("unique_id").size()
            min_series_len = int(series_lengths.min())
            median_series_len = int(series_lengths.median())
            lags = self._get_lags(task.freq, median_series_len, task.seasonality)
            date_features = self._get_date_features(task.freq)
            target_transforms = self._get_target_transforms(task.seasonality, min_series_len)

        forecaster = self._create_mlforecast(task.freq, lags, date_features, target_transforms)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            start_time = time.monotonic()
            forecaster.fit(train_df, static_features=[])
            training_time = time.monotonic() - start_time

            start_time = time.monotonic()
            preds_df = forecaster.predict(window.horizon, X_df=future_df)
            inference_time = time.monotonic() - start_time

        return (
            self._format_predictions(preds_df, task.quantile_levels, window.target_columns),
            training_time,
            inference_time,
        )

    def fit_predict(self, task: fev.Task) -> tuple[list[datasets.DatasetDict], float, float, dict]:
        """Fit and predict on all windows of a task.

        Returns:
            predictions: List of DatasetDict, one per evaluation window
            training_time: Total time spent on fitting (seconds)
            inference_time: Total time spent on predicting (seconds)
            extra_info: Dictionary with model metadata
        """
        task.load_full_dataset()

        predictions = []
        total_training_time = 0.0
        total_inference_time = 0.0

        for window in task.iter_windows():
            preds, train_time, infer_time = self._predict_window(window, task)
            predictions.append(preds)
            total_training_time += train_time
            total_inference_time += infer_time

        extra_info = {"regressor": self.regressor}
        return predictions, total_training_time, total_inference_time, extra_info


class MLForecastAutoModel(MLForecastModel):
    """MLForecast with preprocessing tuning via Optuna."""

    def __init__(
        self,
        regressor: Literal["lightgbm", "catboost"] = "lightgbm",
        num_samples: int = 20,
        n_windows: int = 3,
        hpo_time_limit: float | None = 1800,
        n_jobs: int = 1,
        **kwargs,
    ):
        super().__init__(regressor=regressor, **kwargs)
        self.num_samples = num_samples
        self.n_windows = n_windows
        self.hpo_time_limit = hpo_time_limit
        self.n_jobs = n_jobs

    def _filter_series_for_hpo(self, train_df: pd.DataFrame, horizon: int) -> pd.DataFrame | None:
        min_required = (self.n_windows + 1) * horizon + 1
        series_lengths = train_df.groupby("unique_id").size()
        valid_series = series_lengths[series_lengths >= min_required].index

        min_series_count = min(10, max(1, len(series_lengths) // 10))
        if len(valid_series) < min_series_count:
            return None

        return train_df[train_df["unique_id"].isin(valid_series)]

    def _get_preprocessing_search_space(
        self, seasonality: int, default_lags: list[int], default_date_features: list, min_series_len: int
    ):
        from mlforecast.lag_transforms import ExponentiallyWeightedMean, RollingMean
        from mlforecast.target_transforms import Differences, LocalStandardScaler

        candidate_transforms = [[LocalStandardScaler()]]
        if min_series_len > 1:
            candidate_transforms.append([Differences([1]), LocalStandardScaler()])
        if min_series_len > seasonality:
            candidate_transforms.insert(0, [Differences([seasonality]), LocalStandardScaler()])
        if seasonality > 1 and min_series_len > seasonality + 1:
            candidate_transforms.append([Differences([1, seasonality]), LocalStandardScaler()])

        candidate_lag_transforms = [None, {1: [ExponentiallyWeightedMean(0.9)]}]
        if seasonality > 1:
            candidate_lag_transforms.append({seasonality: [RollingMean(window_size=seasonality, min_samples=1)]})

        def config(trial):
            tfm_idx = trial.suggest_categorical("target_transforms_idx", range(len(candidate_transforms)))
            lag_tfm_idx = trial.suggest_categorical("lag_transforms_idx", range(len(candidate_lag_transforms)))
            return {
                "target_transforms": candidate_transforms[tfm_idx],
                "lags": default_lags,
                "lag_transforms": candidate_lag_transforms[lag_tfm_idx],
                "date_features": default_date_features,
            }

        return config

    def _run_hpo(
        self, train_df: pd.DataFrame, task: fev.Task, lags: list[int], date_features: list, min_series_len: int
    ) -> dict:
        """Run HPO and return the tuned config."""
        import optuna
        from mlforecast.auto import AutoMLForecast, AutoModel

        optuna.logging.set_verbosity(optuna.logging.ERROR)

        forecaster = AutoMLForecast(
            models={self.regressor: AutoModel(model=self._create_model(), config=lambda t: {})},
            freq=task.freq,
            init_config=self._get_preprocessing_search_space(task.seasonality, lags, date_features, min_series_len),
            fit_config=lambda t: {"static_features": []},
        )

        optimize_kwargs = {}
        if self.n_jobs != 1:
            optimize_kwargs["n_jobs"] = self.n_jobs
        if self.hpo_time_limit is not None:
            optimize_kwargs["timeout"] = self.hpo_time_limit

        forecaster.fit(
            train_df,
            n_windows=self.n_windows,
            h=task.horizon,
            num_samples=self.num_samples,
            optimize_kwargs=optimize_kwargs or None,
        )

        best_mlf = forecaster.models_[self.regressor]
        return {
            "lags": list(best_mlf.ts.lags) if best_mlf.ts.lags is not None else [],
            "date_features": best_mlf.ts.date_features if best_mlf.ts.date_features else [],
            "target_transforms": best_mlf.ts.target_transforms if best_mlf.ts.target_transforms else [],
        }

    def _compute_global_min_series_len(self, task: fev.Task) -> int:
        min_len = float("inf")
        for window in task.iter_windows():
            train_df, _ = self._prepare_data(window, task)
            min_len = min(min_len, int(train_df.groupby("unique_id").size().min()))
        return int(min_len)

    def fit_predict(self, task: fev.Task) -> tuple[list[datasets.DatasetDict], float, float, dict]:
        """Fit and predict on all windows of a task.

        Returns:
            predictions: List of DatasetDict, one per evaluation window
            training_time: Total time spent on fitting including HPO (seconds)
            inference_time: Total time spent on predicting (seconds)
            extra_info: Dictionary with model metadata
        """
        task.load_full_dataset()

        min_series_len = self._compute_global_min_series_len(task)

        first_window = task.get_window(0)
        train_df, _ = self._prepare_data(first_window, task)
        hpo_train_df = self._filter_series_for_hpo(train_df, task.horizon)

        tuned_config = None
        hpo_time = 0.0

        if hpo_train_df is not None:
            series_lengths = hpo_train_df.groupby("unique_id").size()
            effective_series_len = int(series_lengths.median()) - self.n_windows * task.horizon
            lags = self._get_lags(task.freq, effective_series_len, task.seasonality)
            date_features = self._get_date_features(task.freq)

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    start_time = time.monotonic()
                    tuned_config = self._run_hpo(hpo_train_df, task, lags, date_features, min_series_len)
                    hpo_time = time.monotonic() - start_time
            except Exception:
                pass

        predictions = []
        total_training_time = hpo_time
        total_inference_time = 0.0

        for window in task.iter_windows():
            preds, train_time, infer_time = self._predict_window(window, task, tuned_config)
            predictions.append(preds)
            total_training_time += train_time
            total_inference_time += infer_time

        return predictions, total_training_time, total_inference_time, {}


if __name__ == "__main__":
    # Configuration
    use_auto = True  # Set to False for fixed preprocessing
    model_name = "lightgbm"  # "lightgbm" or "catboost"
    num_tasks = None  # Set to small number for testing, None for full benchmark

    benchmark = fev.Benchmark.from_yaml(
        "https://raw.githubusercontent.com/autogluon/fev/refs/heads/main/benchmarks/fev_bench/tasks.yaml"
    )

    if use_auto:
        model = MLForecastAutoModel(regressor=model_name)
        model_name = f"auto{model_name}"
    else:
        model = MLForecastModel(regressor=model_name)

    summaries = []
    for task in tqdm(benchmark.tasks[:num_tasks]):
        predictions, training_time, inference_time, extra_info = model.fit_predict(task)
        evaluation_summary = task.evaluation_summary(
            predictions,
            model_name=model_name,
            training_time_s=training_time,
            inference_time_s=inference_time,
            extra_info=extra_info,
        )
        print(evaluation_summary)
        summaries.append(evaluation_summary)

    summary_df = pd.DataFrame(summaries)
    print(summary_df)
    summary_df.to_csv(f"{model_name}.csv", index=False)

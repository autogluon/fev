import pathlib
import warnings
from typing import Callable

import numpy as np
import pandas as pd
import scipy.stats
from packaging.version import parse as parse_version

__all__ = [
    "leaderboard",
    "pairwise_comparison",
    "pivot_table",
]

# Use Arrow dtypes to correctly handle missing values
TASK_DEF_DTYPES = {
    "dataset_path": pd.StringDtype(),
    "dataset_config": pd.StringDtype(),
    "horizon": pd.Int64Dtype(),
    "initial_cutoff": pd.StringDtype(),
    "min_context_length": pd.Int64Dtype(),
    "max_context_length": pd.Int64Dtype(),
    "seasonality": pd.Int64Dtype(),
    "eval_metric": pd.StringDtype(),
    "extra_metrics": pd.StringDtype(),
    "quantile_levels": pd.StringDtype(),
    "id_column": pd.StringDtype(),
    "timestamp_column": pd.StringDtype(),
    "target": pd.StringDtype(),
    "generate_univariate_targets_from": pd.StringDtype(),
    "known_dynamic_columns": pd.StringDtype(),
    "past_dynamic_columns": pd.StringDtype(),
    "static_columns": pd.StringDtype(),
    "task_name": pd.StringDtype(),
}

RESULTS_DTYPES = {
    **TASK_DEF_DTYPES,
    "model_name": pd.StringDtype(),
    "test_error": float,
    "training_time_s": float,
    "trained_on_this_dataset": pd.BooleanDtype(),
    "inference_time_s": float,
    "fev_version": pd.StringDtype(),
}

TASK_DEF_COLUMNS = list(TASK_DEF_DTYPES)
LAST_BREAKING_VERSION = "0.6.0"

# Valid types for summaries
SummaryType = pd.DataFrame | list[dict] | str | pathlib.Path


def _summary_to_df(summary: SummaryType) -> pd.DataFrame:
    """Load a single summary as a pandas DataFrame"""

    if isinstance(summary, pd.DataFrame):
        df = summary
    elif isinstance(summary, list) and isinstance(summary[0], dict):
        df = pd.DataFrame(summary)
    elif isinstance(summary, (str, pathlib.Path)):
        file_path = str(summary)
        try:
            if file_path.endswith(".json"):
                df = pd.read_json(file_path, orient="records")
            elif file_path.endswith(".csv"):
                df = pd.read_csv(file_path)
            else:
                raise ValueError("Path to summaries must ends with '.json' or '.csv'")
        except Exception:
            raise ValueError(f"Unable to load summaries from file '{file_path}.")
    else:
        raise ValueError(
            f"Invalid type of summary {type(summary)}. Expected one of pd.DataFrame, list[dict], str or Path."
        )
    return df


def _load_summaries(summaries: SummaryType | list[SummaryType]) -> pd.DataFrame:
    """Load potentially multiple summary objects into a single pandas DataFrame.

    Ensures that all expected columns are present and have correct dtypes.
    """
    if not isinstance(summaries, list) or (isinstance(summaries, list) and isinstance(summaries[0], dict)):
        summaries = [summaries]
    summaries_df = pd.concat([_summary_to_df(summary) for summary in summaries])

    missing_columns = sorted([col for col in RESULTS_DTYPES if col not in summaries_df])
    if len(missing_columns) > 0:
        warnings.warn(f"Columns {missing_columns} are missing from summaries, filling them with None", stacklevel=3)
    for col in missing_columns:
        summaries_df[col] = None
    summaries_df = summaries_df.astype(RESULTS_DTYPES)
    try:
        min_version = summaries_df["fev_version"].apply(parse_version).min()
        if min_version < parse_version(LAST_BREAKING_VERSION):
            warnings.warn(
                f"Evaluation summaries contain results from fev < {LAST_BREAKING_VERSION}. "
                "Results may not be comparable due to breaking changes.",
                stacklevel=3,
            )
    except Exception:
        raise ValueError(
            "Unable to parse `fev_version` column in the evaluation summaries. "
            "Make sure all summaries are produced by `fev`"
        )
    return summaries_df


def pivot_table(
    summaries: SummaryType | list[SummaryType],
    metric_column: str = "test_error",
    task_columns: str | list[str] = ["dataset_path", "dataset_config"],
    aggfunc: str = "mean",
    baseline_model: str | None = None,
) -> pd.DataFrame:
    """Compute the average score for each model for each task.

    Returns a DataFrame where entry df.iloc[i, j] contains the score of model j on task i.
    """
    summaries = _load_summaries(summaries).astype({metric_column: "float64"})

    pivot_df = summaries.pivot_table(index=task_columns, columns="model_name", values=metric_column, aggfunc=aggfunc)
    if baseline_model is not None:
        if baseline_model not in pivot_df.columns:
            raise ValueError(
                f"baseline_model '{baseline_model}' not found. Available models: {pivot_df.columns.tolist()}"
            )
        pivot_df = pivot_df.divide(pivot_df[baseline_model], axis=0)
    return pivot_df


def _filter_models(
    summaries_df: pd.DataFrame,
    model_column: str = "model_name",
    included_models: list[str] | None = None,
    excluded_models: list[str] | None = None,
) -> pd.DataFrame:
    if excluded_models is not None and included_models is not None:
        raise ValueError("Only one of `excluded_models` and `included_models` can be provided")
    elif excluded_models is not None:
        summaries_df = summaries_df[~summaries_df[model_column].isin(excluded_models)]
    elif included_models is not None:
        summaries_df = summaries_df[summaries_df[model_column].isin(included_models)]
    return summaries_df


def leaderboard(
    summaries: SummaryType | list[SummaryType],
    metric_column: str = "test_error",
    task_columns: str | list[str] = "task_name",
    model_column: str = "model_name",
    baseline_model: str = "SeasonalNaive",
    min_relative_error: float | None = 1e-2,
    max_relative_error: float | None = 100.0,
    remove_failures: bool = False,
    relative_error_for_failures: float | None = None,
    included_models: list[str] | None = None,
    excluded_models: list[str] | None = None,
    n_resamples: int = 1000,
    seed: int = 123,
):
    """Generate a leaderboard with aggregate performance metrics for all models.

    Computes skill score (1 - geometric mean relative error) and win rate with bootstrap confidence
    intervals across all tasks. Models are ranked by skill score.

    Parameters
    ----------
    summaries : SummaryType | list[SummaryType]
        Evaluation summaries as DataFrame, list of dicts, or file path(s)
    metric_column : str, default "test_error"
        Column name containing the metric to evaluate
    task_columns : str | list[str], default "task_name"
        Column(s) defining unique tasks for grouping
    model_column : str, default "model_name"
        Column name containing model identifiers
    baseline_model : str, default "SeasonalNaive"
        Model name to use for relative error computation
    min_relative_error : float, default 1e-3
        Lower bound for clipping relative errors when baseline_model is used
    max_relative_error : float, default 5
        Upper bound for clipping relative errors when baseline_model is used
    remove_failures : bool, default False
        If True, remove tasks where any model failed. Takes precedence over relative_error_for_failures
    included_models : list[str], optional
        Models to include (mutually exclusive with excluded_models)
    excluded_models : list[str], optional
        Models to exclude (mutually exclusive with included_models)
    n_resamples : int, default 1000
        Number of bootstrap samples for confidence intervals
    seed : int, default 123
        Random seed for reproducible bootstrap sampling

    Returns
    -------
    pd.DataFrame
        Leaderboard sorted by skill score, with columns:
        - skill_score: Skill score (1 - geometric mean relative error)
        - skill_score_lower: Lower bound of 95% confidence interval
        - skill_score_upper: Upper bound of 95% confidence interval
        - win_rate: Fraction of pairwise comparisons won against other models
        - win_rate_lower: Lower bound of 95% confidence interval
        - win_rate_upper: Upper bound of 95% confidence interval
        - median_inference_time_s: Median inference time across tasks
    """
    summaries = _load_summaries(summaries)
    summaries = _filter_models(
        summaries, model_column=model_column, included_models=included_models, excluded_models=excluded_models
    )
    errors_df = summaries.pivot_table(index=task_columns, columns=model_column, values=metric_column)
    if baseline_model not in errors_df.columns:
        raise ValueError(
            f"baseline_model '{baseline_model}' is missing. Available models: {errors_df.columns.to_list()}"
        )
    if num_baseline_failures := errors_df[baseline_model].isna().sum():
        raise ValueError(
            f"Results for baseline_model '{baseline_model}' are missing for {num_baseline_failures} tasks."
        )
    errors_df = errors_df.divide(errors_df[baseline_model], axis=0)
    errors_df = errors_df.clip(lower=min_relative_error, upper=max_relative_error)
    if remove_failures:
        errors_df = errors_df.dropna()
        if len(errors_df) == 0:
            raise ValueError("All results are missing for some models.")
    elif relative_error_for_failures is not None:
        errors_df = errors_df.fillna(relative_error_for_failures)
    num_failures_per_model = errors_df.isna().sum()
    if num_failures_per_model.sum():
        raise ValueError(
            f"Results are missing for the following models:\n{num_failures_per_model[num_failures_per_model > 0]}"
        )
    training_time_df = summaries.pivot_table(index=task_columns, columns=model_column, values="training_time_s")
    inference_time_df = summaries.pivot_table(index=task_columns, columns=model_column, values="inference_time_s")
    win_rate, win_rate_lower, win_rate_upper = bootstrap(
        errors_df.to_numpy(), statistic=_win_rate, n_resamples=n_resamples, seed=seed
    )
    skill_score, skill_score_lower, skill_score_upper = bootstrap(
        errors_df.to_numpy(), statistic=_skill_score, n_resamples=n_resamples, seed=seed
    )
    return pd.DataFrame(
        {
            "skill_score": skill_score,
            "skill_score_lower": skill_score_lower,
            "skill_score_upper": skill_score_upper,
            "win_rate": win_rate,
            "win_rate_lower": win_rate_lower,
            "win_rate_upper": win_rate_upper,
            "median_training_time_s": training_time_df.median(),
            "median_inference_time_s": inference_time_df.median(),
            "num_failures": num_failures_per_model,
        },
        index=errors_df.columns,
    ).sort_values(by="skill_score", ascending=False)


def pairwise_comparison(
    summaries: SummaryType | list[SummaryType],
    metric_column: str = "test_error",
    task_columns: str | list[str] = "dataset_path",
    model_column: str = "model_name",
    baseline_model: str = "SeasonalNaive",  # used for imputation if
    min_relative_error: float | None = 1e-2,
    max_relative_error: float | None = 100.0,
    remove_failures: bool = False,
    included_models: list[str] | None = None,
    excluded_models: list[str] | None = None,
    n_resamples: int = 1000,
    seed: int = 123,
) -> pd.DataFrame:
    """Compute pairwise performance comparisons between all model pairs.

    For each pair of models, calculates skill score (1 - geometric mean relative error) and
    win rate with bootstrap confidence intervals across all tasks.

    Parameters
    ----------
    summaries : SummaryType | list[SummaryType]
        Evaluation summaries as DataFrame, list of dicts, or file path(s)
    metric_column : str, default "test_error"
        Column name containing the metric to evaluate
    task_columns : str | list[str], default "dataset_path"
        Column(s) defining unique tasks for grouping
    model_column : str, default "model_name"
        Column name containing model identifiers
    remove_failures : bool, default False
        If True, remove tasks where any model failed; if False, raise error if failures exist
    included_models : list[str], optional
        Models to include (mutually exclusive with excluded_models)
    excluded_models : list[str], optional
        Models to exclude (mutually exclusive with included_models)
    n_resamples : int, default 1000
        Number of bootstrap samples for confidence intervals
    seed : int, default 123
        Random seed for reproducible bootstrap sampling

    Returns
    -------
    pd.DataFrame
        Pairwise comparison results with MultiIndex (model_1, model_2) and columns:
        - skill_score: 1 - geometric mean of model_1/model_2 error ratios
        - skill_score_lower: Lower bound of 95% confidence interval
        - skill_score_upper: Upper bound of 95% confidence interval
        - win_rate: Fraction of tasks where model_1 outperforms model_2
        - win_rate_lower: Lower bound of 95% confidence interval
        - win_rate_upper: Upper bound of 95% confidence interval
    """
    summaries = _load_summaries(summaries)
    summaries = _filter_models(summaries, included_models=included_models, excluded_models=excluded_models)
    errors_df = summaries.pivot_table(index=task_columns, columns=model_column, values=metric_column)
    if remove_failures:
        errors_df = errors_df.dropna()
        if len(errors_df) == 0:
            raise ValueError("All results are missing for some models.")
        print(len(errors_df))
    elif baseline_model is not None:
        if baseline_model not in errors_df.columns:
            raise ValueError(
                f"baseline_model '{baseline_model}' is missing. Available models: {errors_df.columns.to_list()}"
            )
        for col in errors_df.columns:
            if col != baseline_model:
                errors_df[col] = errors_df[col].fillna(errors_df[baseline_model])
    num_failures_per_model = errors_df.isna().sum()
    if num_failures_per_model.sum():
        raise ValueError(
            f"Results are missing for the following models:\n{num_failures_per_model[num_failures_per_model > 0]}"
        )
    model_order = errors_df.rank(axis=1).mean().sort_values().index
    errors_df = errors_df[model_order]
    skill_score, skill_score_lower, skill_score_upper = bootstrap(
        errors_df.to_numpy(),
        statistic=lambda x: _pairwise_skill_score(x, min_relative_error, max_relative_error),
        n_resamples=n_resamples,
        seed=seed,
    )
    win_rate, win_rate_lower, win_rate_upper = bootstrap(
        errors_df.to_numpy(),
        statistic=_pairwise_win_rate,
        n_resamples=n_resamples,
        seed=seed,
    )
    return pd.DataFrame(
        {
            "skill_score": skill_score.flatten(),
            "skill_score_lower": skill_score_lower.flatten(),
            "skill_score_upper": skill_score_upper.flatten(),
            "win_rate": win_rate.flatten(),
            "win_rate_lower": win_rate_lower.flatten(),
            "win_rate_upper": win_rate_upper.flatten(),
        },
        index=pd.MultiIndex.from_product([errors_df.columns, errors_df.columns], names=["model_1", "model_2"]),
    )


def _win_rate(errors: np.ndarray) -> np.ndarray:
    A, B = errors[:, :, None], errors[:, None, :]
    wins = (A < B).mean(0) + 0.5 * (A == B).mean(0)  # [n_models, n_models, ...]
    # Fill diagonal with NaN to avoid counting self-ties as wins
    diag_indices = np.arange(wins.shape[0])
    wins[diag_indices, diag_indices] = float("nan")
    return np.nanmean(wins, axis=1)  # [n_models, ...]


def _skill_score(errors: np.ndarray) -> np.ndarray:
    return 1 - scipy.stats.gmean(errors, axis=0)  # [n_models, ...]


def _pairwise_win_rate(errors: np.ndarray) -> np.ndarray:
    A, B = errors[:, :, None], errors[:, None, :]
    return (A < B).mean(0) + 0.5 * (A == B).mean(0)  # [n_models, n_models, ...]


def _pairwise_skill_score(
    errors: np.ndarray, min_relative_error: float | None = None, max_relative_error: float | None = None
) -> np.ndarray:
    A, B = errors[:, :, None], errors[:, None, :]
    ratios = A / B
    if min_relative_error is not None or max_relative_error is not None:
        ratios = np.clip(ratios, min_relative_error, max_relative_error)
    return 1 - scipy.stats.gmean(ratios, axis=0)  # [n_models, n_models, ...]


def bootstrap(
    errors: np.ndarray,
    statistic: Callable[[np.ndarray], np.ndarray],
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 123,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return point estimate and confidence interval bounds for a statistic.

    Parameters
    ----------
    errors : np.ndarray
        Error of each model on each task, shape [n_tasks, n_models]
    statistic : Callable[[np.ndarray], np.ndarray]
        A function working on numpy arrays [n_tasks, n_models, ...] -> [n_models, ...]
    n_resamples : int, default 1000
        Number of bootstrap samples for confidence intervals
    alpha : float, default 0.05
        Significance level for (1-alpha) confidence intervals
    seed : int, default 123
        Random seed for reproducible bootstrap sampling

    Returns
    -------
    point_estimate : np.ndarray
        Point estimate computed on full data, shape [n_models, ...]
    lower : np.ndarray
        Lower bound of (1-alpha) confidence interval, shape [n_models, ...]
    upper : np.ndarray
        Upper bound of (1-alpha) confidence interval, shape [n_models, ...]
    """
    assert errors.ndim == 2, "errors must have shape [n_tasks, n_models]"
    assert 0 < alpha < 1, "alpha must be in (0, 1)"
    n_tasks, n_models = errors.shape
    point_estimate = statistic(errors)
    assert point_estimate.shape[0] == n_models

    rng = np.random.default_rng(seed=seed)
    indices = rng.integers(0, len(errors), size=(n_resamples, len(errors)))  # [n_resamples, n_tasks]
    errors_resampled = errors[indices, :].transpose(1, 2, 0)  # [n_tasks, n_models, n_resamples]
    output = statistic(errors_resampled)  # [n_models, ..., n_resamples]
    lower = np.quantile(output, 0.5 * alpha, axis=-1)
    upper = np.quantile(output, 1 - 0.5 * alpha, axis=-1)
    return point_estimate, lower, upper

import ast
import pathlib
import warnings
from typing import Callable, Literal, TypeAlias

import numpy as np
import pandas as pd
import scipy.stats
from packaging.version import parse as parse_version

from .constants import DEPRECATED_TASK_FIELDS

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
MODEL_COLUMN = "model_name"

# Valid types for summaries
SummaryType: TypeAlias = pd.DataFrame | list[dict] | str | pathlib.Path


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
                raise ValueError("Path to summaries must end with '.json' or '.csv'")
        except Exception:
            raise ValueError(f"Unable to load summaries from file '{file_path}'.")
    else:
        raise ValueError(
            f"Invalid type of summary {type(summary)}. Expected one of pd.DataFrame, list[dict], str or Path."
        )
    for old_name, new_name in DEPRECATED_TASK_FIELDS.items():
        if old_name in df.columns and new_name in df.columns:
            raise ValueError(
                f"Both deprecated '{old_name}' and '{new_name}' columns are present in the evaluation summary."
            )
    return df.rename(columns=DEPRECATED_TASK_FIELDS)


def _sanitize_quantile_levels(quantile_levels: str | list) -> str:
    if isinstance(quantile_levels, str):
        quantile_levels = ast.literal_eval(quantile_levels)
    if quantile_levels != quantile_levels:  # only true if quantile_levels is NaN
        return "[]"
    if not isinstance(quantile_levels, list):
        raise ValueError(f"Unexpected dtype for quantile_levels: {type(quantile_levels)}")
    return str([round(q, 4) for q in quantile_levels])


def _load_summaries(summaries: SummaryType | list[SummaryType], check_fev_version: bool = False) -> pd.DataFrame:
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
    summaries_df["quantile_levels"] = summaries_df["quantile_levels"].apply(_sanitize_quantile_levels)
    summaries_df = summaries_df.astype(RESULTS_DTYPES)
    if check_fev_version:
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
    task_columns: str | list[str] = TASK_DEF_COLUMNS.copy(),
    baseline_model: str | None = None,
    check_fev_version: bool = False,
) -> pd.DataFrame:
    """Convert evaluation summaries into a pivot table for analysis.

    Creates a matrix where rows represent tasks and columns represent models, with each
    cell containing the specified metric value. Optionally normalizes all scores relative
    to a baseline model.

    Parameters
    ----------
    summaries : SummaryType | list[SummaryType]
        Evaluation summaries as DataFrame, list of dicts, or file path(s)
    metric_column : str, default "test_error"
        Column name containing the metric to pivot
    task_columns : str | list[str], default TASK_DEF_COLUMNS
        Column(s) defining unique tasks. Used as the pivot table index
    baseline_model : str, optional
        If provided, divide all scores by this model's scores to get relative performance
    check_fev_version : bool, default False
        If True, check that fev_version in the summary is >= LAST_BREAKING_VERSION.

    Returns
    -------
    pd.DataFrame
        Pivot table with task combinations as index and model names as columns.
        Values are raw scores or relative scores (if `baseline_model` specified)

    Raises
    ------
    ValueError
        If duplicate model/task combinations exist, or results for `baseline_model` are missing when `baseline_model`
        is provided.
    """
    summaries = _load_summaries(summaries, check_fev_version=check_fev_version).astype({metric_column: "float64"})

    if isinstance(task_columns, str):
        task_columns = [task_columns]
    metric_with_index = summaries.set_index(task_columns + [MODEL_COLUMN])[metric_column]
    duplicates = metric_with_index.index.duplicated()
    if duplicates.any():
        duplicate_indices = metric_with_index.index[duplicates]
        raise ValueError(
            f"Cannot unstack: duplicate index combinations found. First duplicates: {duplicate_indices[:5].tolist()}"
        )
    pivot_df = metric_with_index.unstack()
    if baseline_model is not None:
        if baseline_model not in pivot_df.columns:
            raise ValueError(
                f"baseline_model '{baseline_model}' not found. Available models: {pivot_df.columns.tolist()}"
            )
        pivot_df = pivot_df.divide(pivot_df[baseline_model], axis=0)
        if num_baseline_failures := pivot_df[baseline_model].isna().sum():
            raise ValueError(
                f"Results for baseline_model '{baseline_model}' are missing for "
                f"{num_baseline_failures} out of {len(pivot_df)} tasks."
            )
    return pivot_df


def _filter_models(
    summaries_df: pd.DataFrame,
    included_models: list[str] | None = None,
    excluded_models: list[str] | None = None,
) -> pd.DataFrame:
    if excluded_models is not None and included_models is not None:
        raise ValueError("Only one of `excluded_models` and `included_models` can be provided")
    elif excluded_models is not None:
        summaries_df = summaries_df[~summaries_df[MODEL_COLUMN].isin(excluded_models)]
    elif included_models is not None:
        summaries_df = summaries_df[summaries_df[MODEL_COLUMN].isin(included_models)]
    return summaries_df


def leaderboard(
    summaries: SummaryType | list[SummaryType],
    metric_column: str = "test_error",
    missing_strategy: Literal["error", "drop", "impute"] = "error",
    baseline_model: str = "seasonal_naive",
    min_relative_error: float | None = 1e-2,
    max_relative_error: float | None = 100.0,
    included_models: list[str] | None = None,
    excluded_models: list[str] | None = None,
    n_resamples: int | None = None,
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
    baseline_model : str, default "SeasonalNaive"
        Model name to use for relative error computation
    missing_strategy : Literal["error", "drop", "impute"], default "error"
        How to handle missing results:

        - `"error"`: Raise error if any results are missing
        - `"drop"`: Remove tasks where any model failed
        - `"impute"`: Fill missing results with `baseline_model` scores
    min_relative_error : float, default 1e-2
        Lower bound for clipping relative errors w.r.t. the `baseline_model`
    max_relative_error : float, default 100
        Upper bound for clipping relative errors w.r.t. the `baseline_model`
    included_models : list[str], optional
        Models to include (mutually exclusive with `excluded_models`)
    excluded_models : list[str], optional
        Models to exclude (mutually exclusive with `included_models`)
    n_resamples : int | None, default None
        Number of bootstrap samples for confidence intervals. If None, confidence intervals are not computed
    seed : int, default 123
        Random seed for reproducible bootstrap sampling

    Returns
    -------
    pd.DataFrame
        Leaderboard sorted by `skill_score`, with columns:

        - `skill_score`: Skill score (1 - geometric mean relative error)
        - `skill_score_lower`: Lower bound of 95% confidence interval (only if n_resamples is not None)
        - `skill_score_upper`: Upper bound of 95% confidence interval (only if n_resamples is not None)
        - `win_rate`: Fraction of pairwise comparisons won against other models
        - `win_rate_lower`: Lower bound of 95% confidence interval (only if n_resamples is not None)
        - `win_rate_upper`: Upper bound of 95% confidence interval (only if n_resamples is not None)
        - `median_training_time_s`: Median training time across tasks
        - `median_inference_time_s`: Median inference time across tasks
        - `training_corpus_overlap`: Mean fraction of tasks where model was trained on the dataset
        - `num_failures`: Number of tasks where the model failed
    """
    summaries = _load_summaries(summaries, check_fev_version=True)
    summaries = _filter_models(summaries, included_models=included_models, excluded_models=excluded_models)
    errors_df = pivot_table(summaries, metric_column=metric_column, baseline_model=baseline_model)
    errors_df = errors_df.clip(lower=min_relative_error, upper=max_relative_error)

    num_failures_per_model = errors_df.isna().sum()
    if missing_strategy == "drop":
        errors_df = errors_df.dropna()
        if len(errors_df) == 0:
            raise ValueError("All results are missing for some models.")
        print(f"{len(errors_df)} tasks left after removing failures")
    elif missing_strategy == "impute":
        # For leaderboard, baseline scores are already 1.0 after normalization, so fill with 1.0
        errors_df = errors_df.fillna(1.0)
    elif missing_strategy == "error":
        if num_failures_per_model.sum():
            raise ValueError(
                f"Summaries contain {len(errors_df)} tasks. Results are missing for the following models:"
                f"\n{num_failures_per_model[num_failures_per_model > 0]}"
            )
    else:
        raise ValueError(f"Invalid {missing_strategy=}, expected one of ['error', 'drop', 'impute']")
    bootstrap_resamples = 1 if n_resamples is None else n_resamples
    win_rate, win_rate_lower, win_rate_upper = bootstrap(
        errors_df.to_numpy(), statistic=_win_rate, n_resamples=bootstrap_resamples, seed=seed
    )
    skill_score, skill_score_lower, skill_score_upper = bootstrap(
        errors_df.to_numpy(), statistic=_skill_score, n_resamples=bootstrap_resamples, seed=seed
    )

    training_time_df = pivot_table(summaries, metric_column="training_time_s")
    inference_time_df = pivot_table(summaries, metric_column="inference_time_s")
    training_corpus_overlap_df = pivot_table(summaries, metric_column="trained_on_this_dataset")

    result_df = pd.DataFrame(
        {
            "skill_score": skill_score,
            "skill_score_lower": skill_score_lower,
            "skill_score_upper": skill_score_upper,
            "win_rate": win_rate,
            "win_rate_lower": win_rate_lower,
            "win_rate_upper": win_rate_upper,
            # Select only tasks that are also in errors_df (in case some tasks were dropped with missing_strategy="drop")
            "median_training_time_s": training_time_df.loc[errors_df.index].median(),
            "median_inference_time_s": inference_time_df.loc[errors_df.index].median(),
            "training_corpus_overlap": training_corpus_overlap_df.loc[errors_df.index].mean(),
            "num_failures": num_failures_per_model,
        },
        index=errors_df.columns,
    )
    if n_resamples is None:
        result_df = result_df.drop(
            columns=["skill_score_lower", "skill_score_upper", "win_rate_lower", "win_rate_upper"]
        )
    return result_df.sort_values(by="skill_score", ascending=False)


def pairwise_comparison(
    summaries: SummaryType | list[SummaryType],
    metric_column: str = "test_error",
    missing_strategy: Literal["error", "drop", "impute"] = "error",
    baseline_model: str | None = None,
    min_relative_error: float | None = 1e-2,
    max_relative_error: float | None = 100.0,
    included_models: list[str] | None = None,
    excluded_models: list[str] | None = None,
    n_resamples: int | None = None,
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
    missing_strategy : Literal["error", "drop", "impute"], default "error"
        How to handle missing results:

        - `"error"`: Raise error if any results are missing
        - `"drop"`: Remove tasks where any model failed
        - `"impute"`: Fill missing results with `baseline_model` scores
    baseline_model : str, optional
        Required only when missing_strategy="impute"
    min_relative_error : float, optional, default 1e-2
        Lower bound for clipping error ratios in pairwise comparisons
    max_relative_error : float, optional, default 100.0
        Upper bound for clipping error ratios in pairwise comparisons
    included_models : list[str], optional
        Models to include (mutually exclusive with `excluded_models`)
    excluded_models : list[str], optional
        Models to exclude (mutually exclusive with `included_models`)
    n_resamples : int | None, default None
        Number of bootstrap samples for confidence intervals. If None, confidence intervals are not computed
    seed : int, default 123
        Random seed for reproducible bootstrap sampling

    Returns
    -------
    pd.DataFrame
        Pairwise comparison results with `pd.MultiIndex` `(model_1, model_2)` and columns:

        - `skill_score`: 1 - geometric mean of `model_1/model_2` error ratios
        - `skill_score_lower`: Lower bound of 95% confidence interval (only if n_resamples is not None)
        - `skill_score_upper`: Upper bound of 95% confidence interval (only if n_resamples is not None)
        - `win_rate`: Fraction of tasks where `model_1` outperforms `model_2`
        - `win_rate_lower`: Lower bound of 95% confidence interval (only if n_resamples is not None)
        - `win_rate_upper`: Upper bound of 95% confidence interval (only if n_resamples is not None)
    """
    summaries = _load_summaries(summaries, check_fev_version=True)
    summaries = _filter_models(summaries, included_models=included_models, excluded_models=excluded_models)
    errors_df = pivot_table(summaries, metric_column=metric_column)
    num_failures_per_model = errors_df.isna().sum()

    if missing_strategy == "drop":
        errors_df = errors_df.dropna()
        if len(errors_df) == 0:
            raise ValueError("All results are missing for some models.")
        print(f"{len(errors_df)} tasks left after removing failures")
    elif missing_strategy == "impute":
        if baseline_model is None:
            raise ValueError("baseline_model is required when missing_strategy='impute'")
        if baseline_model not in errors_df.columns:
            raise ValueError(
                f"baseline_model '{baseline_model}' is missing. Available models: {errors_df.columns.to_list()}"
            )
        for col in errors_df.columns:
            if col != baseline_model:
                errors_df[col] = errors_df[col].fillna(errors_df[baseline_model])
    elif missing_strategy == "error":
        if num_failures_per_model.sum():
            raise ValueError(
                f"Summaries contain {len(errors_df)} tasks. Results are missing for the following models:"
                f"\n{num_failures_per_model[num_failures_per_model > 0]}"
            )
    else:
        raise ValueError(f"Invalid {missing_strategy=}, expected one of ['error', 'drop', 'impute']")
    model_order = errors_df.rank(axis=1).mean().sort_values().index
    errors_df = errors_df[model_order]

    bootstrap_resamples = 1 if n_resamples is None else n_resamples
    skill_score, skill_score_lower, skill_score_upper = bootstrap(
        errors_df.to_numpy(),
        statistic=lambda x: _pairwise_skill_score(x, min_relative_error, max_relative_error),
        n_resamples=bootstrap_resamples,
        seed=seed,
    )
    win_rate, win_rate_lower, win_rate_upper = bootstrap(
        errors_df.to_numpy(),
        statistic=_pairwise_win_rate,
        n_resamples=bootstrap_resamples,
        seed=seed,
    )

    result_df = pd.DataFrame(
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
    if n_resamples is None:
        result_df = result_df.drop(
            columns=["skill_score_lower", "skill_score_upper", "win_rate_lower", "win_rate_upper"]
        )
    return result_df


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
    errors_resampled = errors[indices].transpose(1, 2, 0)  # [n_tasks, n_models, n_resamples]
    output = statistic(errors_resampled)  # [n_models, ..., n_resamples]
    lower = np.quantile(output, 0.5 * alpha, axis=-1)
    upper = np.quantile(output, 1 - 0.5 * alpha, axis=-1)
    return point_estimate, lower, upper


# Methods that can be used as `statistic` in `bootstrap`. Expect `errors` with shape [n_tasks, n_models, n_resamples]
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

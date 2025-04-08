import pathlib
import pprint
import warnings

import pandas as pd
from scipy.stats import gmean

from fev.benchmark import Benchmark

# Use Arrow dtypes to correctly handle missing values
TASK_DEF_DTYPES = {
    "dataset_path": pd.StringDtype(),
    "dataset_config": pd.StringDtype(),
    "horizon": pd.Int64Dtype(),
    "cutoff": pd.StringDtype(),
    "lead_time": pd.Int64Dtype(),
    "min_ts_length": pd.Int64Dtype(),
    "max_context_length": pd.Int64Dtype(),
    "seasonality": pd.Int64Dtype(),
    "eval_metric": pd.StringDtype(),
    "quantile_levels": pd.StringDtype(),
    "id_column": pd.StringDtype(),
    "timestamp_column": pd.StringDtype(),
    "target_column": pd.StringDtype(),
    "multiple_target_columns": pd.StringDtype(),
    "past_dynamic_columns": pd.StringDtype(),
    "excluded_columns": pd.StringDtype(),
}

RESULTS_DTYPES = {
    **TASK_DEF_DTYPES,
    "model_name": pd.StringDtype(),
    "test_error": float,
    "training_time_s": float,
    "trained_on_this_dataset": pd.BooleanDtype(),
    "inference_time_s": float,
}

TASK_DEF_COLUMNS = list(TASK_DEF_DTYPES)

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

    for col in RESULTS_DTYPES:
        if col not in summaries_df:
            warnings.warn(f"Column '{col}' is missing from summaries, filling with None", stacklevel=3)
            summaries_df[col] = None
    return summaries_df.astype(RESULTS_DTYPES)


def pivot_table(
    summaries: SummaryType | list[SummaryType],
    metric_column: str = "test_error",
    task_columns: str | list[str] = "dataset_name",
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


def leaderboard(
    summaries: SummaryType | list[SummaryType],
    metric_column: str = "test_error",
    baseline_model: str = "seasonal_naive",
    min_relative_error: float = 1e-3,
    max_relative_error: float = 5,
    remove_failures: bool = True,
    rel_score_failures: float | None = None,
    included_models: list[str] | None = None,
    excluded_models: list[str] | None = None,
    benchmark: Benchmark | None = None,
    validate_dataset_fingerprints: bool = False,
):
    """Summarize benchmark results into a single table reporting aggregate performance of each model.

    For each task, we compute the relative error of each model as `test_error[model] / test_error[baseline_model]`.
    Relative scores are clipped in the range `[min_relative_error, max_relative_error]`, and all model failures
    are replaced with `max_relative_error`.

    For each model, the following metrics are reported:

    - `gmean_relative_error` - geometric mean of relative scores across all tasks
    - `avg_rank` - average rank across all tasks
    - `avg_inference_time_s` - average inference time of each model (in seconds)
    - `median_inference_time_s` - median inference time of each model (in seconds)
    - `avg_training_time_s` - average training time of each model (in seconds)
    - `median_training_time_s` - median training time of each model (in seconds)
    - `training_corpus_overlap` - fraction of the datasets used in the benchmark that were included in the model's training corpus
    - `num_failures` - number of tasks for which each model failed

    Parameters
    ----------
    summaries : pd.DataFrame | list[dict] | str | list[pd.DataFrame] | list[list[dict]] | list[str]
        One or multiple summary objects containing evaluation results.
        Each summary object can be represented as a:

        - list of dictionaries produced by :meth:`fev.Task.evaluation_summary`
        - a DataFrame where each row corresponds to the evaluation summary on one task
        - path to a JSON (orient="records") or CSV file with the evaluation summaries
    baseline_model : str, default "seasonal_naive"
        Name of the baseline model that is used to compute relative scores.
    min_relative_error : float, default 1e-3
        Relative scores below this value are clipped.
    max_relative_error : float, default 5
        Relative scores above this value are clipped.
    remove_failures : bool, default True
        If True, tasks where at least one model failed will be excluded from the summaries. If False, relative scores
        for failed tasks will be replaced with `max_relative_error`.
    rel_score_failures : float, optional
        If provided and `remove_failures=False`, relative scores for failed models will be replaced with this value.
    included_models : list[str], optional
        If provided, only results for these models will be considered.
    excluded_models : list[str], optional
        If provided, results for these models will be excluded. Cannot be used if `included_models` is also provided.
    benchmark : fev.Benchmark, optional
        If provided, the results will be computed using only the tasks included in the benchmark. If results for some
        tasks of this benchmark are missing, an exception will be raised.
    validate_dataset_fingerprints : bool, default False
        If `True`, this method will assert that the dataset fingerprint is unique for each task. This ensures that
        the same dataset version was used by all models.
    """
    summaries = _load_summaries(summaries).astype({metric_column: "float64"}).set_index(TASK_DEF_COLUMNS)

    if validate_dataset_fingerprints:
        num_fingerprints_per_task = summaries.groupby(TASK_DEF_COLUMNS, dropna=False)["dataset_fingerprint"].nunique()
        tasks_with_different_fingerprints = num_fingerprints_per_task.index[num_fingerprints_per_task > 1]
        if len(tasks_with_different_fingerprints) > 0:
            raise ValueError(
                f"{len(tasks_with_different_fingerprints)} tasks have different dataset fingerprints:\n"
                f"{tasks_with_different_fingerprints.to_frame(index=False)}"
            )

    if benchmark is not None:
        expected_tasks = pd.MultiIndex.from_frame(
            pd.DataFrame([task.to_dict() for task in benchmark.tasks]).astype(TASK_DEF_DTYPES)[TASK_DEF_COLUMNS]
        )
        available_tasks = summaries.index.drop_duplicates()
        missing_tasks = expected_tasks.difference(available_tasks).to_frame().to_dict(orient="records")
        if len(missing_tasks):
            raise ValueError(
                f"Missing results for {len(missing_tasks)} tasks:\n{pprint.pformat(missing_tasks, sort_dicts=False)}"
            )
        summaries = summaries.loc[expected_tasks]
    summaries = summaries.set_index(["model_name"], append=True)

    if excluded_models is not None and included_models is not None:
        raise ValueError("Only one of `excluded_models` and `included_models` can be provided")
    elif excluded_models is not None:
        summaries = summaries.query("model_name not in @excluded_models")
    elif included_models is not None:
        summaries = summaries.query("model_name in @included_models")

    error_per_model = summaries[metric_column].unstack()
    if baseline_model not in error_per_model.columns:
        raise ValueError(
            f"baseline_model '{baseline_model}' not found. Available models: {error_per_model.columns.tolist()}"
        )
    if error_per_model[baseline_model].isna().any():
        missing_baseline = error_per_model[error_per_model[baseline_model].isna()].index
        raise ValueError(
            f"baseline_model must have valid test_error for all tasks. Missing tasks:\n{missing_baseline}"
        )

    num_failures_per_model = error_per_model.isna().sum()
    rel_error_per_model = error_per_model.divide(error_per_model[baseline_model], axis=0).clip(
        lower=min_relative_error, upper=max_relative_error
    )
    inference_time_s_per_model = summaries["inference_time_s"].unstack()
    training_time_s_per_model = summaries["training_time_s"].unstack()
    trained_on_this_dataset = summaries["trained_on_this_dataset"].unstack()
    if remove_failures:
        rel_error_per_model = rel_error_per_model.dropna(how="any")
        if len(rel_error_per_model) < len(error_per_model):
            print(f"Keeping {len(rel_error_per_model)} / {len(error_per_model)} tasks where no model failed.")
        inference_time_s_per_model = inference_time_s_per_model.reindex(rel_error_per_model.index)
        training_time_s_per_model = training_time_s_per_model.reindex(rel_error_per_model.index)
        trained_on_this_dataset = trained_on_this_dataset.reindex(rel_error_per_model.index)
    else:
        rel_error_per_model = rel_error_per_model.fillna(rel_score_failures or max_relative_error)
    avg_rank_per_model = error_per_model.rank(axis=1).mean()

    agg_scores = pd.concat(
        {
            "gmean_relative_error": rel_error_per_model.apply(gmean),
            "avg_rank": avg_rank_per_model,
            "avg_inference_time_s": inference_time_s_per_model.mean(),
            "median_inference_time_s": inference_time_s_per_model.median(),
            "avg_training_time_s": training_time_s_per_model.mean(),
            "median_training_time_s": training_time_s_per_model.median(),
            "training_corpus_overlap": trained_on_this_dataset.mean(),
            "num_failures": num_failures_per_model,
        },
        axis=1,
    )
    return agg_scores.sort_values(by="gmean_relative_error")

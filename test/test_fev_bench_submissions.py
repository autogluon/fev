"""Sanity checks for fev-bench leaderboard submissions.

A submission is a results CSV in ``benchmarks/fev_bench/results/`` (one file per model family;
a file may hold several model sizes). These tests run in CI whenever ``benchmarks/`` is modified
and verify that submissions are formatted correctly: right task-definition columns, valid task
names, no duplicates, and that they slot into the leaderboard without breaking metric
aggregation. They are intentionally lightweight and offline.
"""

from pathlib import Path

import pandas as pd
import pytest

import fev
from fev.analysis import TASK_DEF_COLUMNS, pivot_table

REPO_ROOT = Path(__file__).resolve().parent.parent
FEV_BENCH_DIR = REPO_ROOT / "benchmarks" / "fev_bench"
RESULTS_DIR = FEV_BENCH_DIR / "results"
TASKS_YAML = FEV_BENCH_DIR / "tasks.yaml"

# Metrics reported on the leaderboard that every submission must provide.
REQUIRED_METRICS = ["MASE", "SQL"]
# Task-definition columns feed the pivot; a malformed one silently splits a task into extra rows.
REQUIRED_COLUMNS = TASK_DEF_COLUMNS + ["model_name", "task_name", *REQUIRED_METRICS]

# Reject pathological files before parsing. Legitimate submissions are well under 1 MB.
MAX_RESULT_FILE_BYTES = 5 * 1024 * 1024

RESULT_FILES = sorted(RESULTS_DIR.glob("*.csv"))


@pytest.fixture(scope="module")
def benchmark_task_names() -> set[str]:
    return {task.task_name for task in fev.Benchmark.from_yaml(str(TASKS_YAML)).tasks}


@pytest.mark.parametrize("result_file", RESULT_FILES, ids=lambda p: p.name)
def test_when_submission_added_then_it_is_valid(result_file: Path, benchmark_task_names: set[str]):
    assert result_file.stat().st_size <= MAX_RESULT_FILE_BYTES, (
        f"{result_file.name} is unexpectedly large (> {MAX_RESULT_FILE_BYTES} bytes)"
    )

    # Parsing untrusted submission data: read as plain data, never evaluated as code.
    summary = pd.read_csv(result_file)

    missing_columns = sorted(col for col in REQUIRED_COLUMNS if col not in summary.columns)
    assert not missing_columns, f"{result_file.name} is missing required columns: {missing_columns}"

    # A submission may report several models (e.g. model sizes), but not duplicate (model, task) rows.
    duplicates = summary[summary.duplicated(["model_name", "task_name"])]
    assert duplicates.empty, (
        f"{result_file.name} has duplicate (model_name, task_name) rows: "
        f"{duplicates[['model_name', 'task_name']].values.tolist()[:5]}"
    )

    # Every reported task must belong to the benchmark. Omitting tasks is allowed (imputed downstream).
    extra_tasks = sorted(set(summary["task_name"]) - benchmark_task_names)
    assert not extra_tasks, f"{result_file.name} reports tasks not in fev-bench: {extra_tasks}"

    for metric in REQUIRED_METRICS:
        n_missing = summary[metric].isna().sum()
        assert n_missing == 0, f"{result_file.name} has {n_missing} missing/NaN values in '{metric}'"


@pytest.mark.parametrize("metric", REQUIRED_METRICS)
def test_when_all_submissions_loaded_then_leaderboard_covers_every_task(metric: str, benchmark_task_names: set[str]):
    summaries = pd.concat([pd.read_csv(f) for f in RESULT_FILES], ignore_index=True)

    # Task-definition columns must be correct: the union of all submissions must pivot to exactly one
    # row per benchmark task. An extra row here means a task-def column is malformed in some submission.
    pivot = pivot_table(summaries, metric_column=metric)
    assert len(pivot) == len(benchmark_task_names), (
        f"Pivot for '{metric}' has {len(pivot)} tasks, expected {len(benchmark_task_names)}"
    )

    lb = fev.analysis.leaderboard(
        summaries=summaries,
        metric_column=metric,
        missing_strategy="impute",
        baseline_model="Seasonal Naive",
        leakage_imputation_model="Chronos-Bolt",
    )
    assert set(summaries["model_name"]) == set(lb.index)

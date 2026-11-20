"""Sanity checks for fev-bench leaderboard submissions.

A submission is a results CSV in ``benchmarks/fev_bench/results/`` plus its entry in
``benchmarks/fev_bench/models.yaml``. These tests run in CI whenever ``benchmarks/`` is modified
and verify that submissions are formatted correctly: right task-definition columns, valid task
names, no duplicates, valid metadata, and that they slot into the leaderboard without breaking
metric aggregation. They are intentionally lightweight and offline.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest
from packaging.requirements import InvalidRequirement, Requirement

import fev
from fev.analysis import TASK_DEF_COLUMNS, pivot_table

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from fev_bench_metadata import load_model_metadata  # noqa: E402

FEV_BENCH_DIR = REPO_ROOT / "benchmarks" / "fev_bench"
RESULTS_DIR = FEV_BENCH_DIR / "results"
TASKS_YAML = FEV_BENCH_DIR / "tasks.yaml"

# Metrics reported on the leaderboard that every submission must provide.
REQUIRED_METRICS = ["MASE", "SQL"]
# Runtime columns the leaderboard reports. Values are wall-clock seconds for the whole system.
TIME_COLUMNS = ["training_time_s", "inference_time_s"]
# Model types that fit inside `predict` and may legitimately report zero training time.
TYPES_WITHOUT_TRAINING_STEP = {"statistical"}
# Task-definition columns feed the pivot; a malformed one silently splits a task into extra rows.
# description_fingerprint is optional: summaries without it are filled with None on load.
REQUIRED_COLUMNS = [c for c in TASK_DEF_COLUMNS if c != "description_fingerprint"] + [
    "model_name",
    "task_name",
    *REQUIRED_METRICS,
    *TIME_COLUMNS,
]

# Reject pathological files before parsing. Legitimate submissions are well under 1 MB.
MAX_RESULT_FILE_BYTES = 5 * 1024 * 1024

RESULT_FILES = sorted(RESULTS_DIR.glob("*.csv"))

MODELS_DIR = REPO_ROOT / "models"
WRAPPER_DIRS = sorted(p for p in MODELS_DIR.iterdir() if p.is_dir() and p.name != "__pycache__")
# Wrappers merged before exact pins were required. Do not add new entries.
UNPINNED_REQUIREMENTS_ALLOWLIST = {"flowstate", "sundial", "timesfm-2.5", "timesfm-3", "tirex-2"}


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


@pytest.mark.parametrize("model_name", sorted(load_model_metadata()))
def test_when_submission_added_then_it_reports_runtimes(model_name: str):
    metadata = load_model_metadata()[model_name]
    summary = pd.read_csv(RESULTS_DIR / metadata["results_file"])
    for column in TIME_COLUMNS:
        values = pd.to_numeric(summary[column], errors="coerce")
        bad = summary.loc[~(values >= 0) | values.isin([float("inf")]), "task_name"].tolist()
        assert not bad, f"{model_name}: '{column}' must be a finite non-negative number, invalid for tasks {bad[:5]}"

    no_inference = summary.loc[summary["inference_time_s"] <= 0, "task_name"].tolist()
    assert not no_inference, f"{model_name}: 'inference_time_s' must be > 0, got 0 for tasks {no_inference[:5]}"

    if not metadata["zero_shot"] and metadata["model_type"] not in TYPES_WITHOUT_TRAINING_STEP:
        no_training = summary.loc[summary["training_time_s"] <= 0, "task_name"].tolist()
        assert not no_training, (
            f"{model_name} is trained on each task (zero_shot: false) but reports 'training_time_s' <= 0 "
            f"for tasks {no_training[:5]}. Report the full per-task training time of the whole system."
        )


def test_when_submission_added_then_it_has_valid_metadata():
    # Validates every models.yaml entry and that entries and results files match one-to-one.
    load_model_metadata()


@pytest.mark.parametrize("wrapper_dir", WRAPPER_DIRS, ids=lambda p: p.name)
def test_when_wrapper_added_then_it_is_a_single_model_file(wrapper_dir: Path):
    # Model code belongs in a released package. The wrapper is one model.py adapter plus requirements.txt.
    assert (wrapper_dir / "model.py").is_file(), f"models/{wrapper_dir.name}/ is missing model.py"
    assert (wrapper_dir / "requirements.txt").is_file(), f"models/{wrapper_dir.name}/ is missing requirements.txt"
    extra = sorted(
        str(path.relative_to(wrapper_dir))
        for pattern in ("*.py", "*.json")
        for path in wrapper_dir.rglob(pattern)
        if path.name != "model.py" or path.parent != wrapper_dir
    )
    assert not extra, (
        f"models/{wrapper_dir.name}/ must contain a single model.py with no other .py or .json files, found {extra}. "
        "Move the model code into a package released on PyPI and pin it in requirements.txt."
    )


def _is_exact_pin(requirement: Requirement) -> bool:
    if requirement.url:
        # VCS URLs must reference a commit or tag. Archive URLs point to a fixed file.
        return not requirement.url.startswith("git+") or "@" in requirement.url.split("://", 1)[-1]
    specifiers = list(requirement.specifier)
    return len(specifiers) == 1 and specifiers[0].operator in ("==", "===") and "*" not in specifiers[0].version


@pytest.mark.parametrize("wrapper_dir", WRAPPER_DIRS, ids=lambda p: p.name)
def test_when_wrapper_added_then_requirements_are_pinned(wrapper_dir: Path):
    if wrapper_dir.name in UNPINNED_REQUIREMENTS_ALLOWLIST:
        pytest.skip("Merged before exact pins were required")
    unpinned = []
    for line in (wrapper_dir / "requirements.txt").read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        try:
            requirement = Requirement(line)
        except InvalidRequirement:
            unpinned.append(line)
            continue
        if not _is_exact_pin(requirement):
            unpinned.append(line)
    assert not unpinned, (
        f"models/{wrapper_dir.name}/requirements.txt must pin every dependency exactly "
        f"(`pkg==X.Y.Z`, or a git URL with `@<commit or tag>`), found {unpinned}"
    )

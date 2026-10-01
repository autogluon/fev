"""Sanity checks for fev-bench leaderboard submissions.

A submission is a results CSV in ``benchmarks/fev_bench/results/`` (exactly one model per file)
plus its entry in ``benchmarks/fev_bench/models.yaml``. These tests run in CI whenever
``benchmarks/`` is modified and verify that submissions are formatted correctly: right
task-definition columns, valid task names, no duplicates, complete display metadata, and that they
slot into the leaderboard without breaking metric aggregation. They are intentionally lightweight
and offline.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

import fev
from fev.analysis import TASK_DEF_COLUMNS, pivot_table

REPO_ROOT_FOR_IMPORT = Path(__file__).resolve().parent.parent
# The metadata loader is shared with the leaderboard app, which reads it straight from a fev clone.
sys.path.insert(0, str(REPO_ROOT_FOR_IMPORT / "scripts"))

from fev_bench_metadata import (  # noqa: E402
    DEFAULT_MODELS_YAML,
    load_model_metadata,
    load_models_yaml,
    read_model_names,
)

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

    # One model per file keeps the models.yaml key (the file name) pointing at a single leaderboard row.
    reported_models = sorted(set(summary["model_name"]))
    assert len(reported_models) == 1, (
        f"{result_file.name} reports {len(reported_models)} models ({reported_models}), expected exactly 1. "
        "Split the file so that each results CSV holds a single model."
    )

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


def test_when_submission_added_then_it_has_valid_metadata():
    """Every results file has a models.yaml entry (and vice versa) that satisfies the schema.

    ``load_model_metadata`` validates the schema and both directions of the file <-> entry mapping,
    so the leaderboard can never show a model with missing, stale, or malformed metadata.
    """
    metadata = load_model_metadata(RESULTS_DIR, DEFAULT_MODELS_YAML)
    assert set(metadata) == set(read_model_names(RESULTS_DIR).values())


def test_when_models_yaml_parsed_then_keys_match_results_file_names():
    # Guards against a typo'd key being reported as an unrelated missing/orphaned entry.
    assert set(load_models_yaml(DEFAULT_MODELS_YAML)) == {f.stem for f in RESULT_FILES}


VALID_METADATA_ENTRY = {
    "organization": "Acme",
    "url": "https://example.com",
    "model_type": "pretrained",
    "zero_shot": True,
    "commercial_use": True,
}

# Each case is a models.yaml entry that must be rejected, so a malformed submission fails CI instead
# of silently showing wrong metadata on the leaderboard. `None` means "drop this required field".
INVALID_METADATA_ENTRIES = {
    "missing_required_field": {"commercial_use": None},
    "unknown_field": {"model_size": "small"},
    "invalid_model_type": {"model_type": "api"},
    "non_bool_commercial_use": {"commercial_use": "yes"},
    "non_bool_hidden": {"hidden": "true"},
    "empty_organization": {"organization": "  "},
}


@pytest.mark.parametrize("override", INVALID_METADATA_ENTRIES.values(), ids=INVALID_METADATA_ENTRIES.keys())
def test_when_metadata_entry_is_invalid_then_loading_raises(override: dict, tmp_path: Path):
    entry = {**VALID_METADATA_ENTRY, **override}
    entry = {k: v for k, v in entry.items() if v is not None}
    models_yaml = tmp_path / "models.yaml"
    models_yaml.write_text(yaml.safe_dump({"my-model": entry}))
    with pytest.raises(ValueError):
        load_models_yaml(models_yaml)


def test_when_results_file_has_no_metadata_entry_then_loading_raises(tmp_path: Path):
    models_yaml = tmp_path / "models.yaml"
    models_yaml.write_text(yaml.safe_dump({"my-model": VALID_METADATA_ENTRY}))
    with pytest.raises(ValueError, match="without a models.yaml entry"):
        load_model_metadata(RESULTS_DIR, models_yaml)

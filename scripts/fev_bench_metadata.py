"""Load and validate the fev-bench model metadata in ``benchmarks/fev_bench/models.yaml``.

This is the single source of truth for everything the leaderboard shows about a model that is not
computed from its results: organization, link, model type, zero-shot and commercial-use flags. The
leaderboard app (``save_tables.py`` in the HF Space repo) and the submission tests both read it
through this module, so a submission only ever declares its metadata once.

Entries are keyed by the file name of the results CSV (without ``.csv``), and every results file
holds exactly one model. ``load_model_metadata`` re-keys entries by ``model_name`` for consumers
that join against summaries.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import yaml

FEV_BENCH_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "fev_bench"
DEFAULT_MODELS_YAML = FEV_BENCH_DIR / "models.yaml"
DEFAULT_RESULTS_DIR = FEV_BENCH_DIR / "results"

# Model types a submission may declare. Documented for submitters in models.yaml.
MODEL_TYPES = ("pretrained", "task-specific", "statistical", "system", "closed-api")

# Types shown on the leaderboard unless the reader opts into more. Results for the remaining types
# either cannot be independently reproduced (closed-api) or are not comparable to single models
# (system), so they are opt-in rather than filtered out.
DEFAULT_VISIBLE_MODEL_TYPES = ("pretrained", "task-specific", "statistical")

REQUIRED_FIELDS = ("organization", "url", "model_type", "zero_shot", "commercial_use")
# Optional fields and their defaults. `display_name` defaults to the model_name from the results CSV.
OPTIONAL_FIELDS: dict[str, Any] = {"display_name": None, "hidden": False}


def _validate_entry(key: str, entry: Any) -> dict[str, Any]:
    """Check one models.yaml entry against the schema and fill in optional-field defaults."""
    if not isinstance(entry, dict):
        raise ValueError(f"models.yaml entry '{key}' must be a mapping, got {type(entry).__name__}")

    missing = [f for f in REQUIRED_FIELDS if f not in entry]
    if missing:
        raise ValueError(f"models.yaml entry '{key}' is missing required fields: {missing}")

    unknown = sorted(set(entry) - set(REQUIRED_FIELDS) - set(OPTIONAL_FIELDS))
    if unknown:
        raise ValueError(
            f"models.yaml entry '{key}' has unknown fields: {unknown}. "
            f"Allowed: {sorted({*REQUIRED_FIELDS, *OPTIONAL_FIELDS})}"
        )

    entry = {**OPTIONAL_FIELDS, **entry}

    if entry["model_type"] not in MODEL_TYPES:
        raise ValueError(
            f"models.yaml entry '{key}' has invalid model_type '{entry['model_type']}'. Allowed: {list(MODEL_TYPES)}"
        )
    # Booleans must really be booleans: YAML happily parses "no"/"false" into strings when quoted,
    # and a truthy string would silently flip the meaning of these flags on the leaderboard.
    for field in ("zero_shot", "commercial_use", "hidden"):
        if not isinstance(entry[field], bool):
            raise ValueError(f"models.yaml entry '{key}' field '{field}' must be true or false, got {entry[field]!r}")
    for field in ("organization", "url"):
        if not isinstance(entry[field], str) or not entry[field].strip():
            raise ValueError(f"models.yaml entry '{key}' field '{field}' must be a non-empty string")

    return entry


def load_models_yaml(models_yaml: Path = DEFAULT_MODELS_YAML) -> dict[str, dict[str, Any]]:
    """Parse and validate models.yaml. Returns entries keyed by results-CSV file name (no suffix)."""
    content = yaml.safe_load(models_yaml.read_text())
    if not isinstance(content, dict):
        raise ValueError(f"{models_yaml} must contain a mapping of results-file name -> metadata")
    return {key: _validate_entry(key, entry) for key, entry in content.items()}


def read_model_names(results_dir: Path = DEFAULT_RESULTS_DIR) -> dict[str, str]:
    """Map each results CSV's file name (no suffix) to the single model_name it reports.

    Raises if a file reports zero or several model names, or if two files report the same name:
    both would silently merge or split rows when the summaries are concatenated.
    """
    model_names: dict[str, str] = {}
    for path in sorted(results_dir.glob("*.csv")):
        with path.open(newline="") as f:
            names = {row["model_name"] for row in csv.DictReader(f)}
        if len(names) != 1:
            raise ValueError(
                f"{path.name} reports {len(names)} model names ({sorted(names)}), expected exactly 1. "
                "Split the file so that each results CSV holds a single model."
            )
        model_names[path.stem] = names.pop()

    duplicates = {
        name: sorted(stem for stem, n in model_names.items() if n == name)
        for name in {n for n in model_names.values() if list(model_names.values()).count(n) > 1}
    }
    if duplicates:
        raise ValueError(f"Several results files report the same model_name: {duplicates}")

    return model_names


def load_model_metadata(
    results_dir: Path = DEFAULT_RESULTS_DIR,
    models_yaml: Path = DEFAULT_MODELS_YAML,
) -> dict[str, dict[str, Any]]:
    """Return validated metadata keyed by ``model_name``, checked against the results files.

    Raises if a results file has no models.yaml entry or an entry has no results file, so the
    leaderboard can never display a model with missing or stale metadata.
    """
    metadata = load_models_yaml(models_yaml)
    model_names = read_model_names(results_dir)

    missing_entries = sorted(set(model_names) - set(metadata))
    if missing_entries:
        raise ValueError(
            f"Results files without a models.yaml entry: {missing_entries}. "
            f"Add one entry per file to {models_yaml.name}."
        )
    orphaned_entries = sorted(set(metadata) - set(model_names))
    if orphaned_entries:
        raise ValueError(
            f"models.yaml entries without a results file in {results_dir.name}/: {orphaned_entries}. "
            "Remove the entry or add the missing results CSV."
        )

    return {
        model_name: {
            **entry,
            "model_name": model_name,
            "display_name": entry["display_name"] or model_name,
            "results_file": f"{stem}.csv",
        }
        for stem, model_name in model_names.items()
        for entry in [metadata[stem]]
    }


def hidden_model_names(
    results_dir: Path = DEFAULT_RESULTS_DIR,
    models_yaml: Path = DEFAULT_MODELS_YAML,
) -> list[str]:
    """Model names flagged ``hidden: true``, i.e. excluded from the leaderboard and paper figures."""
    metadata = load_model_metadata(results_dir, models_yaml)
    return sorted(name for name, entry in metadata.items() if entry["hidden"])

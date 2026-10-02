"""Leaderboard metadata for fev-bench submissions, stored in ``benchmarks/fev_bench/models.yaml``.

``ModelMetadata`` below is the reference for the ``models.yaml`` format. The submission tests and the
leaderboard app both load the file through ``load_model_metadata``.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

FEV_BENCH_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "fev_bench"
DEFAULT_MODELS_YAML = FEV_BENCH_DIR / "models.yaml"
DEFAULT_RESULTS_DIR = FEV_BENCH_DIR / "results"


class ModelMetadata(BaseModel):
    """One ``models.yaml`` entry, keyed by the file name of its results CSV (without ``.csv``).

    Each results CSV in ``results/`` holds exactly one model and has exactly one entry.
    """

    # `protected_namespaces=()` allows the `model_type` field name, which pydantic reserves by default.
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True, protected_namespaces=())

    # Who built the model. Use "—" for models provided by a library.
    organization: str = Field(min_length=1)
    # HF model page, GitHub repo, or API docs. Link somewhere the license is visible.
    url: str = Field(min_length=1)
    # If several apply, use the last one listed. The leaderboard hides `system` and `closed-api`
    # models until the reader opts in.
    #   pretrained     open-weights foundation model pretrained on large amounts of data
    #   task-specific  traditional ML model trained from scratch on each task
    #   statistical    classical statistical method, fit per task
    #   system         pipeline combining or selecting several models (AutoML, ensembles)
    #   closed-api     any forecasting component runs behind a closed API, or weights are not public
    model_type: Literal["pretrained", "task-specific", "statistical", "system", "closed-api"]
    # Whether the model forecasts without being trained on the target task.
    zero_shot: bool
    # Whether the model's license permits commercial use.
    commercial_use: bool
    # Only needed if the `model_name` in the results CSV does not display well.
    display_name: str | None = None
    # Set by maintainers to drop a model from the leaderboard and figures (e.g. redundant sizes).
    hidden: bool = False


def load_models_yaml(models_yaml: Path = DEFAULT_MODELS_YAML) -> dict[str, ModelMetadata]:
    """Parse and validate ``models.yaml``, keyed by results-CSV file name (without ``.csv``)."""
    content = yaml.safe_load(models_yaml.read_text())
    if not isinstance(content, dict):
        raise ValueError(f"{models_yaml} must contain a mapping of results-file name -> metadata")
    entries = {}
    for key, entry in content.items():
        try:
            entries[key] = ModelMetadata.model_validate(entry)
        except ValidationError as e:
            raise ValueError(f"Invalid models.yaml entry '{key}':\n{e}") from e
    return entries


def read_model_names(results_dir: Path = DEFAULT_RESULTS_DIR) -> dict[str, str]:
    """Map each results CSV's file name (without ``.csv``) to the single ``model_name`` it reports."""
    model_names = {}
    for path in sorted(results_dir.glob("*.csv")):
        with path.open(newline="") as f:
            names = {row["model_name"] for row in csv.DictReader(f)}
        if len(names) != 1:
            raise ValueError(f"{path.name} reports {len(names)} model names ({sorted(names)}), expected exactly 1")
        model_names[path.stem] = names.pop()

    duplicates = sorted(name for name, count in Counter(model_names.values()).items() if count > 1)
    if duplicates:
        raise ValueError(f"Several results files report the same model_name: {duplicates}")
    return model_names


def load_model_metadata(
    results_dir: Path = DEFAULT_RESULTS_DIR,
    models_yaml: Path = DEFAULT_MODELS_YAML,
) -> dict[str, dict[str, Any]]:
    """Validated metadata keyed by ``model_name``, checked to match the results files one-to-one."""
    metadata = load_models_yaml(models_yaml)
    model_names = read_model_names(results_dir)

    missing_entries = sorted(set(model_names) - set(metadata))
    if missing_entries:
        raise ValueError(f"Results files without a models.yaml entry: {missing_entries}")
    orphaned_entries = sorted(set(metadata) - set(model_names))
    if orphaned_entries:
        raise ValueError(f"models.yaml entries without a results file: {orphaned_entries}")

    return {
        model_name: {
            **metadata[stem].model_dump(),
            "model_name": model_name,
            "display_name": metadata[stem].display_name or model_name,
            "results_file": f"{stem}.csv",
        }
        for stem, model_name in model_names.items()
    }


def hidden_model_names(
    results_dir: Path = DEFAULT_RESULTS_DIR,
    models_yaml: Path = DEFAULT_MODELS_YAML,
) -> list[str]:
    """Names of models flagged ``hidden: true``."""
    metadata = load_model_metadata(results_dir, models_yaml)
    return sorted(name for name, entry in metadata.items() if entry["hidden"])

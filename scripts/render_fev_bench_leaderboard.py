# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "fev",
#     "pandas",
#     "pyyaml",
# ]
# ///
"""Render the fev-bench overall leaderboard as Markdown for the submission-check PR comment.

Builds top-K leaderboard tables for MASE and SQL from the committed summaries in
``benchmarks/fev_bench/results`` (the same ``fev.analysis.leaderboard`` the real leaderboard uses)
and writes a Markdown snippet suitable for posting as a GitHub PR comment.

    uv run scripts/render_fev_bench_leaderboard.py --output table.md --changed citras-fm.csv

The optional ``--changed`` files (result CSVs touched by the PR) have their models marked with a
bold arrow in the table so reviewers can spot the new submission at a glance.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pandas as pd

import fev

BASELINE_MODEL = "Seasonal Naive"
LEAKAGE_IMPUTATION_MODEL = "Chronos-Bolt"
METRICS = ["MASE", "SQL"]
TOP_K = 15

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "benchmarks" / "fev_bench" / "results"

# Leaderboard column -> Markdown header. Order defines the rendered columns.
DISPLAY_COLUMNS = {
    "model_name": "Model",
    "win_rate": "Win rate (%)",
    "skill_score": "Skill score (%)",
    "median_e2e_time_s_per100": "Runtime / 100 series (s)",
    "leakage_pct": "Leakage (%)",
    "failed_pct": "Failed tasks (%)",
}


def _escape(value: str) -> str:
    """Neutralize Markdown/pipe characters so untrusted model names can't break the table."""
    return str(value).replace("|", "\\|").replace("`", "").replace("\n", " ").strip()


def compute_leaderboard(summaries: pd.DataFrame, metric: str) -> pd.DataFrame:
    n_tasks = summaries["task_name"].nunique()
    with warnings.catch_warnings():
        # Zero-shot models have all-NaN training time; nan-aggregating an empty slice warns benignly.
        warnings.simplefilter("ignore", RuntimeWarning)
        lb = fev.analysis.leaderboard(
            summaries=summaries,
            metric_column=metric,
            missing_strategy="impute",
            baseline_model=BASELINE_MODEL,
            leakage_imputation_model=LEAKAGE_IMPUTATION_MODEL,
            normalize_time_per_n_forecasts=100,
        )
    lb = lb.sort_values("win_rate", ascending=False).reset_index()
    lb["win_rate"] = lb["win_rate"] * 100
    lb["skill_score"] = lb["skill_score"] * 100
    lb["leakage_pct"] = lb["training_corpus_overlap"] * 100
    lb["failed_pct"] = lb["num_failures"] / n_tasks * 100
    return lb


def render_table(lb: pd.DataFrame, changed_models: set[str]) -> str:
    lb = lb.head(TOP_K)
    header = "| " + " | ".join(DISPLAY_COLUMNS.values()) + " |"
    separator = "| " + " | ".join("---" for _ in DISPLAY_COLUMNS) + " |"
    rows = []
    for _, row in lb.iterrows():
        name = _escape(row["model_name"])
        if row["model_name"] in changed_models:
            name = f"**{name}** :arrow_left:"
        cells = [name]
        for col in list(DISPLAY_COLUMNS)[1:]:
            cells.append(f"{row[col]:.1f}")
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, separator, *rows])


def changed_models_from_files(files: list[str]) -> set[str]:
    models: set[str] = set()
    for name in files:
        path = RESULTS_DIR / Path(name).name
        if path.exists():
            models.update(pd.read_csv(path)["model_name"].unique().tolist())
    return models


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="Path to write the Markdown table.")
    parser.add_argument(
        "--changed",
        nargs="*",
        default=[],
        help="Result CSV filenames touched by the PR; their models are highlighted.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    files = sorted(RESULTS_DIR.glob("*.csv"))
    summaries = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    changed_models = changed_models_from_files(args.changed)

    sections = [f"## fev-bench leaderboard (top {TOP_K})", ""]
    if changed_models:
        sections.append(f"Models in this PR: {', '.join(sorted(_escape(m) for m in changed_models))} :arrow_left:")
        sections.append("")
    for metric in METRICS:
        lb = compute_leaderboard(summaries, metric)
        sections.append(f"### {metric}")
        sections.append(render_table(lb, changed_models))
        sections.append("")

    args.output.write_text("\n".join(sections))
    print(f"Wrote leaderboard tables to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

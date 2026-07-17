# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "fev",
#     "pandas",
#     "pyyaml",
# ]
# ///
"""Render the fev-bench overall leaderboard as Markdown for the submission-check PR comment.

Reuses the leaderboard computation from ``generate_fev_bench_figures.py`` (the same
``fev.analysis.leaderboard`` the real leaderboard uses) and formats the top-K rows as a Markdown
table.

    uv run scripts/render_fev_bench_leaderboard.py --output table.md --changed citras-fm.csv

The optional ``--changed`` files (result CSVs touched by the PR) have their models highlighted so
reviewers can spot the new submission at a glance.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from generate_fev_bench_figures import DEFAULT_RESULTS_DIR, compute_leaderboard, load_summaries

METRICS = ["MASE", "SQL"]
TOP_K = 15

# Leaderboard column -> Markdown header. Order defines the rendered columns.
DISPLAY_COLUMNS = {
    "model_name": "Model",
    "win_rate": "Win rate (%)",
    "skill_score": "Skill score (%)",
    "median_e2e_time_s_per100": "Runtime / 100 series (s)",
    "leakage_pct": "Leakage (%)",
    "num_failures": "Failed tasks (%)",
}


def _escape(value: str) -> str:
    """Neutralize Markdown/pipe characters so untrusted model names can't break the table."""
    return str(value).replace("|", "\\|").replace("`", "").replace("\n", " ").strip()


def render_table(leaderboard_df: pd.DataFrame, changed_models: set[str]) -> str:
    df = leaderboard_df.head(TOP_K).copy()
    df["leakage_pct"] = df["training_corpus_overlap"] * 100

    header = "| " + " | ".join(DISPLAY_COLUMNS.values()) + " |"
    separator = "| " + " | ".join("---" for _ in DISPLAY_COLUMNS) + " |"
    rows = []
    for _, row in df.iterrows():
        name = _escape(row["model_name"])
        if row["model_name"] in changed_models:
            name = f"**{name}** :arrow_left:"
        cells = [name] + [f"{row[col]:.1f}" for col in list(DISPLAY_COLUMNS)[1:]]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, separator, *rows])


def changed_models_from_files(files: list[str]) -> set[str]:
    models: set[str] = set()
    for name in files:
        path = DEFAULT_RESULTS_DIR / Path(name).name
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
    summaries = load_summaries(DEFAULT_RESULTS_DIR)
    changed_models = changed_models_from_files(args.changed)

    sections = [f"## fev-bench leaderboard (top {TOP_K})", ""]
    if changed_models:
        sections += [f"Models in this PR: {', '.join(sorted(_escape(m) for m in changed_models))} :arrow_left:", ""]
    for metric in METRICS:
        sections += [f"### {metric}", render_table(compute_leaderboard(summaries, metric), changed_models), ""]

    args.output.write_text("\n".join(sections))
    print(f"Wrote leaderboard tables to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

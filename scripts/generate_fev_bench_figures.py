# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "fev",
#     "altair",
#     "vl-convert-python",
#     "pandas",
#     "pyyaml",
#     "requests",
# ]
# ///
"""Generate consistent leaderboard tables and pairwise-comparison figures from fev summaries.

This produces the same leaderboard and pairwise-comparison numbers shown on the fev leaderboard
(https://huggingface.co/spaces/autogluon/fev-leaderboard), so authors can drop consistent figures
into papers and reports. The numbers come straight from ``fev.analysis.leaderboard`` /
``fev.analysis.pairwise_comparison`` (the same functions the leaderboard uses). The pairwise
figures are publication-quality Altair heatmaps (large fonts, color legend, inline 95% CIs);
``construct_pairwise_chart`` also accepts flags to reproduce the leaderboard's interactive look.

Run with ``uv run`` so the inline dependencies above are installed automatically:

    # Full benchmark, every metric, from the committed summaries
    uv run scripts/generate_fev_bench_figures.py

    # A single metric, restricted to the fev-bench-mini subset
    uv run scripts/generate_fev_bench_figures.py --metric SQL \\
        --benchmark benchmarks/fev_bench/tasks_mini.yaml

    # Restrict to an explicit list of tasks
    uv run scripts/generate_fev_bench_figures.py --tasks ETT_1H solar_1D walmart

    # Make the pairwise heatmaps larger
    uv run scripts/generate_fev_bench_figures.py --metric SQL --fig-width 1100

By default summaries are read from ``benchmarks/fev_bench/results``. Outputs land in
``--out-dir`` (default ``figures/``) as vector PDFs:

    leaderboard_<metric>.csv           leaderboard table (raw values)
    leaderboard_<metric>.tex           leaderboard table (paper-style LaTeX)
    pairwise_win_rate_<metric>.pdf     pairwise win-rate heatmap
    pairwise_skill_score_<metric>.pdf  pairwise skill-score heatmap
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import fev

# --- Benchmark configuration (kept in sync with the fev leaderboard) ------------------------

BASELINE_MODEL = "Seasonal Naive"
LEAKAGE_IMPUTATION_MODEL = "Chronos-Bolt"
AVAILABLE_METRICS = ["SQL", "MASE", "WQL", "WAPE"]
SORT_COL = "win_rate"
N_RESAMPLES_FOR_CI = 1000
TOP_K_MODELS_TO_PLOT = 15

# --- Figure curation (script-only; the leaderboard intentionally shows all models) ----------

# Redundant Toto-2.0 sizes dropped by default to avoid crowding figures with one model family.
# Keeps Toto-2.0-22m and Toto-2.0-2.5B. Override with --exclude or disable with --keep-all-models.
DEFAULT_EXCLUDED_MODELS = ["Toto-2.0-4m", "Toto-2.0-313m", "Toto-2.0-1B"]

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "fev_bench" / "results"

# --- Plotting -------------------------------------------------------------------------------

HEATMAP_COLOR_SCHEME = "purplegreen"

# Per-metric heatmap config: (colorbar label, color domain, midpoint, white-text condition)
PAIRWISE_CHART_CONFIG = {
    "win_rate": ("Win Rate", [0, 100], 50, "abs(datum.{col} - 50) > 30"),
    "skill_score": ("Skill Score", [-30, 30], 0, "abs(datum.{col}) > 20"),
}


def _format_ci_value(val: float) -> str:
    """Compact CI number: drop decimals once the magnitude needs 3+ digits."""
    return f"{int(round(val))}" if abs(int(val)) >= 100 else f"{val:.1f}"


def construct_pairwise_chart(
    df: pd.DataFrame,
    col: str,
    *,
    title: str | None = None,
    width: int = 900,
    inline_ci: bool = True,
    show_legend: bool = True,
):
    """Altair heatmap of pairwise win rate / skill score.

    Defaults produce a publication-quality static figure (large fonts, color legend, inline 95% CI
    text under each value, square aspect ratio). The same function powers the interactive
    leaderboard view by passing ``inline_ci=False`` (CI shown via tooltip) and ``title=<metric>``.

    Parameters
    ----------
    df
        Output of ``fev.analysis.pairwise_comparison(...).reset_index()`` with ``model_1``,
        ``model_2``, ``<col>``, ``<col>_lower``, ``<col>_upper`` columns.
    col
        Either ``"win_rate"`` or ``"skill_score"``.
    title
        Optional chart title. Omitted by default (papers use a figure caption instead).
    width
        Chart width in pixels; height is derived to keep the matrix square.
    inline_ci
        If True, render the 95% CI under each cell value (best for static figures). If False,
        the CI is only available via the hover tooltip (best for the interactive leaderboard).
    show_legend
        Whether to draw the color legend.
    """
    import altair as alt

    cbar_label, domain, domain_mid, text_condition_tmpl = PAIRWISE_CHART_CONFIG[col]
    text_condition = text_condition_tmpl.format(col=col)

    df = df.copy()
    for c in [col, f"{col}_lower", f"{col}_upper"]:
        df[c] *= 100
    df["ci_text"] = df.apply(
        lambda row: f"({_format_ci_value(row[f'{col}_lower'])}, {_format_ci_value(row[f'{col}_upper'])})",
        axis=1,
    )

    model_order = df.groupby("model_1")[col].mean().sort_values(ascending=False).index.tolist()
    n_rows, n_cols = df["model_1"].nunique(), df["model_2"].nunique()

    tooltip = [
        alt.Tooltip("model_1:N", title="Model 1"),
        alt.Tooltip("model_2:N", title="Model 2"),
        alt.Tooltip(f"{col}:Q", title=cbar_label.split(" ")[0], format=".1f"),
        alt.Tooltip(f"{col}_lower:Q", title="95% CI Lower", format=".1f"),
        alt.Tooltip(f"{col}_upper:Q", title="95% CI Upper", format=".1f"),
    ]

    base = alt.Chart(df).encode(
        x=alt.X("model_2:N", sort=model_order, title="", axis=alt.Axis(orient="top", labelAngle=-90)),
        y=alt.Y("model_1:N", sort=model_order, title=""),
    )

    legend = alt.Legend(title=f"{cbar_label} (%)", direction="vertical") if show_legend else None
    heatmap = base.mark_rect().encode(
        color=alt.Color(
            f"{col}:Q",
            legend=legend,
            scale=alt.Scale(scheme=HEATMAP_COLOR_SCHEME, domain=domain, domainMid=domain_mid, clamp=True),
        ),
        tooltip=tooltip,
    )

    text_color = alt.condition(text_condition, alt.value("white"), alt.value("black"))
    text_main = base.mark_text(dy=-8, fontSize=10, baseline="top", yOffset=0).encode(
        text=alt.Text(f"{col}:Q", format=".1f"), color=text_color, tooltip=tooltip
    )
    layers = heatmap + text_main
    if inline_ci:
        text_ci = base.mark_text(dy=-8, fontSize=7.5, baseline="top", yOffset=10).encode(
            text=alt.Text("ci_text:N"), color=text_color, tooltip=tooltip
        )
        layers = layers + text_ci

    chart = layers.properties(height=int(width * n_rows / n_cols), width=width)
    if title is not None:
        chart = chart.properties(title={"text": title, "fontSize": 18})
    return (
        chart.configure_axis(titleFontSize=16, labelFontSize=14)
        .configure_title(fontSize=18)
        .configure_legend(gradientLength=90)
        .resolve_scale(color="independent")
    )


# --- Summary loading and metric computation -------------------------------------------------


def load_summaries(results_dir: Path) -> pd.DataFrame:
    """Load and concatenate all summary CSVs from a directory."""
    csv_files = sorted(results_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {results_dir}")
    print(f"Loading {len(csv_files)} summary files from {results_dir}")
    return pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)


def resolve_task_filter(benchmark: str | None, tasks: list[str] | None) -> list[str] | None:
    """Return the list of task_names to keep, or None to keep all tasks."""
    if benchmark is not None:
        task_names = [t.task_name for t in fev.Benchmark.from_yaml(benchmark).tasks]
        print(f"Filtering to {len(task_names)} tasks from benchmark: {benchmark}")
        return task_names
    if tasks:
        print(f"Filtering to {len(tasks)} explicitly provided tasks")
        return tasks
    return None


def filter_summaries(summaries: pd.DataFrame, task_names: list[str] | None) -> pd.DataFrame:
    if task_names is None:
        return summaries
    missing = sorted(set(task_names) - set(summaries["task_name"]))
    if missing:
        raise ValueError(f"{len(missing)} requested tasks not found in summaries: {missing}")
    filtered = summaries[summaries["task_name"].isin(task_names)]
    print(f"  {filtered['task_name'].nunique()} tasks matched in summaries")
    return filtered


def compute_leaderboard(summaries: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    lb = fev.analysis.leaderboard(
        summaries=summaries,
        metric_column=metric_name,
        missing_strategy="impute",
        baseline_model=BASELINE_MODEL,
        leakage_imputation_model=LEAKAGE_IMPUTATION_MODEL,
        normalize_time_per_n_forecasts=100,
    )
    lb = lb.astype("float64").reset_index()
    lb["skill_score"] = lb["skill_score"] * 100
    lb["win_rate"] = lb["win_rate"] * 100
    lb["num_failures"] = lb["num_failures"] / summaries["task_name"].nunique() * 100
    return lb.sort_values(by=SORT_COL, ascending=False)


# Columns kept in the LaTeX leaderboard table, mapped to their (bold) display headers.
LATEX_COLUMNS = {
    "model_name": "Model",
    "win_rate": r"Avg. win rate (\%)",
    "skill_score": r"Skill score (\%)",
    "median_e2e_time_s_per100": "Median runtime / 100 series (s)",
    "training_corpus_overlap": r"Leakage (\%)",
    "num_failures": r"Failed tasks (\%)",
}


def leaderboard_to_latex(leaderboard_df: pd.DataFrame) -> str:
    """Render a paper-style LaTeX leaderboard table from a ``compute_leaderboard`` result."""
    df = leaderboard_df[list(LATEX_COLUMNS)].copy()
    for col in ["win_rate", "skill_score", "median_e2e_time_s_per100", "training_corpus_overlap", "num_failures"]:
        df[col] = df[col].round(1)
    df = df.astype({"training_corpus_overlap": "int"})
    df = df.rename(columns={col: r"\textbf{" + name + r"}" for col, name in LATEX_COLUMNS.items()})
    return df.to_latex(index=False, float_format="%.1f")


def compute_pairwise(summaries: pd.DataFrame, metric_name: str, included_models: list[str]) -> pd.DataFrame:
    # Baseline and leakage-imputation models must be present for imputation, even if outside the top-K
    included_models = included_models + [
        m for m in (BASELINE_MODEL, LEAKAGE_IMPUTATION_MODEL) if m not in included_models
    ]
    return (
        fev.analysis.pairwise_comparison(
            summaries,
            included_models=included_models,
            metric_column=metric_name,
            baseline_model=BASELINE_MODEL,
            missing_strategy="impute",
            n_resamples=N_RESAMPLES_FOR_CI,
            leakage_imputation_model=LEAKAGE_IMPUTATION_MODEL,
        )
        .round(3)
        .reset_index()
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--metric",
        choices=AVAILABLE_METRICS,
        action="append",
        help=f"Metric(s) to generate (repeatable). Default: all of {AVAILABLE_METRICS}.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing the summary CSV files (default: benchmarks/fev_bench/results).",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--benchmark",
        default=None,
        help="Path or URL of a benchmark YAML; only its tasks are included.",
    )
    group.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Explicit list of task_names to include (alternative to --benchmark).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("figures"),
        help="Directory to write tables and figures into (default: figures/).",
    )
    parser.add_argument(
        "--fig-width",
        type=int,
        default=900,
        help="Pairwise figure width in pixels; height scales to keep the matrix square (default: 900).",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=None,
        help=f"Model names to drop from tables/figures (default: {DEFAULT_EXCLUDED_MODELS}).",
    )
    parser.add_argument(
        "--keep-all-models",
        action="store_true",
        help="Disable the default model exclusion and include every model in the summaries.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    metrics = args.metric or AVAILABLE_METRICS

    summaries = load_summaries(args.results_dir)
    task_names = resolve_task_filter(args.benchmark, args.tasks)
    summaries = filter_summaries(summaries, task_names)

    excluded = [] if args.keep_all_models else (args.exclude if args.exclude is not None else DEFAULT_EXCLUDED_MODELS)
    if excluded:
        present = sorted(set(excluded) & set(summaries["model_name"]))
        summaries = summaries[~summaries["model_name"].isin(excluded)]
        print(f"Excluding {len(present)} models: {present}")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for metric in metrics:
        print(f"\nProcessing metric: {metric}")
        leaderboard_df = compute_leaderboard(summaries, metric)
        leaderboard_path = out_dir / f"leaderboard_{metric}.csv"
        leaderboard_df.to_csv(leaderboard_path, index=False)
        print(f"  Saved: {leaderboard_path}")

        latex_path = out_dir / f"leaderboard_{metric}.tex"
        latex_path.write_text(leaderboard_to_latex(leaderboard_df))
        print(f"  Saved: {latex_path}")

        top_models = leaderboard_df.head(TOP_K_MODELS_TO_PLOT)["model_name"].tolist()
        pairwise_df = compute_pairwise(summaries, metric, top_models)

        for col in ["win_rate", "skill_score"]:
            chart = construct_pairwise_chart(pairwise_df, col=col, width=args.fig_width)
            fig_path = out_dir / f"pairwise_{col}_{metric}.pdf"
            chart.save(fig_path)
            print(f"  Saved: {fig_path}")

    print(f"\nAll outputs written to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

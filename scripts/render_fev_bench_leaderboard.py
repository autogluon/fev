# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "fev",
#     "pandas",
#     "pyyaml",
# ]
# ///
"""Print the fev-bench leaderboard (top rows) so reviewers can eyeball a submission in CI logs.

Reuses the leaderboard computation from ``generate_fev_bench_figures.py`` (the same
``fev.analysis.leaderboard`` the real leaderboard uses).

    uv run scripts/render_fev_bench_leaderboard.py
"""

from __future__ import annotations

from generate_fev_bench_figures import DEFAULT_RESULTS_DIR, compute_leaderboard, load_summaries

METRICS = ["MASE", "SQL"]
TOP_K = 15
COLUMNS = [
    "model_name",
    "win_rate",
    "skill_score",
    "median_e2e_time_s_per100",
    "training_corpus_overlap",
    "num_failures",
]


def main() -> int:
    summaries = load_summaries(DEFAULT_RESULTS_DIR)
    for metric in METRICS:
        lb = compute_leaderboard(summaries, metric).head(TOP_K)
        lb["training_corpus_overlap"] = lb["training_corpus_overlap"] * 100  # fraction -> %
        print(f"\n### {metric} (top {TOP_K})\n")
        print(lb[COLUMNS].to_string(index=False, float_format="%.1f"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

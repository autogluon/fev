import pytest

import fev

SUMMARIES_URLS = [
    "https://raw.githubusercontent.com/autogluon/fev/refs/heads/main/benchmarks/chronos_zeroshot/results/chronos_bolt_small.csv",
    "https://raw.githubusercontent.com/autogluon/fev/refs/heads/main/benchmarks/chronos_zeroshot/results/auto_arima.csv",
    "https://raw.githubusercontent.com/autogluon/fev/refs/heads/main/benchmarks/chronos_zeroshot/results/tirex.csv",
    "https://raw.githubusercontent.com/autogluon/fev/refs/heads/main/benchmarks/chronos_zeroshot/results/seasonal_naive.csv",
]


@pytest.mark.parametrize("n_resamples", [None, 1000])
def test_when_leaderboard_called_then_all_expected_columns_are_present(n_resamples):
    expected_columns = [
        "win_rate",
        "win_rate_lower",
        "win_rate_upper",
        "skill_score",
        "skill_score_lower",
        "skill_score_upper",
        "median_training_time_s",
        "median_inference_time_s",
        "training_corpus_overlap",
        "num_failures",
    ]
    if n_resamples is None:
        for col in ["win_rate_lower", "win_rate_upper", "skill_score_lower", "skill_score_upper"]:
            expected_columns.remove(col)
    leaderboard = fev.leaderboard(SUMMARIES_URLS, n_resamples=n_resamples, baseline_model="seasonal_naive")
    assert leaderboard.columns.to_list() == expected_columns


@pytest.mark.parametrize("n_resamples", [None, 1000])
def test_when_pairwise_comparison_called_then_all_expected_columns_are_present(n_resamples):
    expected_columns = [
        "win_rate",
        "win_rate_lower",
        "win_rate_upper",
        "skill_score",
        "skill_score_lower",
        "skill_score_upper",
    ]
    if n_resamples is None:
        for col in ["win_rate_lower", "win_rate_upper", "skill_score_lower", "skill_score_upper"]:
            expected_columns.remove(col)
    pairwise_comparison = fev.pairwise_comparison(SUMMARIES_URLS, n_resamples=n_resamples)
    assert pairwise_comparison.columns.to_list() == expected_columns


def test_when_pivot_table_called_then_errors_df_has_expected_shape():
    summaries = fev.analysis._load_summaries(SUMMARIES_URLS)
    pivot_table = fev.pivot_table(SUMMARIES_URLS)
    assert pivot_table.shape == (summaries["dataset_config"].nunique(), summaries["model_name"].nunique())

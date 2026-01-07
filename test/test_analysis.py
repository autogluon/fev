import pandas as pd
import pytest

import fev
from fev.analysis import RESULTS_DTYPES

SUMMARIES_URLS = [
    "https://raw.githubusercontent.com/autogluon/fev/refs/heads/main/benchmarks/chronos_zeroshot/results/chronos_bolt_small.csv",
    "https://raw.githubusercontent.com/autogluon/fev/refs/heads/main/benchmarks/chronos_zeroshot/results/auto_arima.csv",
    "https://raw.githubusercontent.com/autogluon/fev/refs/heads/main/benchmarks/chronos_zeroshot/results/tirex.csv",
    "https://raw.githubusercontent.com/autogluon/fev/refs/heads/main/benchmarks/chronos_zeroshot/results/seasonal_naive.csv",
]


@pytest.fixture
def mock_summaries():
    base = {k: None for k in RESULTS_DTYPES}
    base.update({"quantile_levels": "[]", "fev_version": "0.7.0", "trained_on_this_dataset": False})
    return pd.DataFrame(
        [
            {
                **base,
                "dataset_path": "d1",
                "model_name": "model_a",
                "test_error": 1.0,
                "training_time_s": 10.0,
                "inference_time_s": 5.0,
                "num_forecasts": 100,
            },
            {
                **base,
                "dataset_path": "d1",
                "model_name": "model_b",
                "test_error": 2.0,
                "training_time_s": 20.0,
                "inference_time_s": 10.0,
                "num_forecasts": 100,
            },
            {
                **base,
                "dataset_path": "d2",
                "model_name": "model_a",
                "test_error": 1.5,
                "training_time_s": 30.0,
                "inference_time_s": 15.0,
                "num_forecasts": 200,
            },
            {
                **base,
                "dataset_path": "d2",
                "model_name": "model_b",
                "test_error": 3.0,
                "training_time_s": 40.0,
                "inference_time_s": 20.0,
                "num_forecasts": 200,
            },
        ]
    )


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


@pytest.mark.parametrize("normalize_time_per_n_forecasts", [None, 100, 1000])
def test_when_leaderboard_called_with_num_forecasts_then_times_are_normalized(
    mock_summaries, normalize_time_per_n_forecasts
):
    result = fev.leaderboard(
        mock_summaries, baseline_model="model_b", normalize_time_per_n_forecasts=normalize_time_per_n_forecasts
    )

    if normalize_time_per_n_forecasts is None:
        # No normalization: median of raw times
        assert result.loc["model_a", "median_training_time_s"] == 20.0  # median(10, 30)
        assert result.loc["model_a", "median_inference_time_s"] == 10.0  # median(5, 15)
    else:
        # Normalized: (time / num_forecasts) * normalize_time_per_n_forecasts
        # model_a: task1 = 10/100*N=0.1N, task2 = 30/200*N=0.15N -> median = 0.125N
        expected_training = 0.125 * normalize_time_per_n_forecasts
        expected_inference = 0.0625 * normalize_time_per_n_forecasts  # median(5/100, 15/200) = median(0.05, 0.075)
        time_suffix = "" if normalize_time_per_n_forecasts is None else f"_per{normalize_time_per_n_forecasts}"
        assert result.loc["model_a", "median_training_time_s" + time_suffix] == expected_training
        assert result.loc["model_a", "median_inference_time_s" + time_suffix] == expected_inference


def test_when_leaderboard_called_with_partial_num_forecasts_then_bfill_works(mock_summaries):
    mock_summaries.loc[mock_summaries["model_name"] == "model_b", "num_forecasts"] = None
    normalize_time_per_n_forecasts = 100
    result = fev.leaderboard(
        mock_summaries.head(2), baseline_model="model_b", normalize_time_per_n_forecasts=normalize_time_per_n_forecasts
    )
    time_suffix = f"_per{normalize_time_per_n_forecasts}"
    assert result.loc["model_a", "median_training_time_s" + time_suffix] == 10.0
    assert result.loc["model_b", "median_training_time_s" + time_suffix] == 20.0


def test_when_leaderboard_called_with_inconsistent_num_forecasts_then_raises(mock_summaries):
    mock_summaries.loc[mock_summaries["model_name"] == "model_b", "num_forecasts"] = 200
    with pytest.raises(ValueError, match="inconsistent values"):
        fev.leaderboard(mock_summaries.head(2), baseline_model="model_b", normalize_time_per_n_forecasts=100)

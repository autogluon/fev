import datasets
import numpy as np
import pandas as pd
import pytest
from autogluon.timeseries import TimeSeriesPredictor

import fev
from fev.metrics import AVAILABLE_METRICS, _seasonal_error_per_item


# Include datasets with NaN values (nn5) and all-zero history values (covid deaths)
@pytest.fixture(scope="module", params=["nn5", "monash_covid_deaths"])
def model_setup(tmp_path_factory, request):
    def _to_pandas(ds: datasets.Dataset) -> pd.DataFrame:
        return ds.to_pandas().explode(["timestamp", "target"]).infer_objects()

    task = fev.Task(
        dataset_path="autogluon/chronos_datasets",
        dataset_config=request.param,
        horizon=59,
        seasonality=3,
        quantile_levels=[0.1, 0.5, 0.9],
    )
    window = task.get_window(0)
    train, future = window.get_input_data()
    test = window.get_ground_truth()

    train_df = _to_pandas(train).rename(columns={"id": "item_id"})
    test_df = _to_pandas(test).rename(columns={"id": "item_id"})

    predictor = TimeSeriesPredictor(
        prediction_length=task.horizon,
        eval_metric_seasonal_period=task.seasonality,
        quantile_levels=task.quantile_levels,
        path=tmp_path_factory.mktemp("predictor"),
    ).fit(train_df, hyperparameters={"SeasonalNaive": {}}, verbosity=0)

    return task, train_df, test_df, predictor


def _fev_predictions(predictor, train_df):
    """Build fev-formatted predictions (one dict per item) from an AutoGluon predictor."""
    ag_predictions = predictor.predict(train_df).rename(columns={"mean": "predictions"})
    return [pred.to_dict("list") for _, pred in ag_predictions.groupby("item_id", as_index=False)]


@pytest.mark.parametrize("eval_metric", list(AVAILABLE_METRICS))
def test_when_metrics_computed_then_score_matches_autogluon(model_setup, eval_metric):
    task, train_df, test_df, predictor = model_setup
    task.eval_metric = eval_metric

    full_df = pd.concat([train_df, test_df])
    if task.eval_metric == "MQL":
        # MQL metric not implemented in AG
        ag_score = predictor.evaluate(full_df, metrics=["WQL"])["WQL"] * -1
        ag_score *= test_df["target"].abs().mean()
    else:
        ag_score = predictor.evaluate(full_df, metrics=[task.eval_metric])[task.eval_metric] * -1

    fev_score = task.evaluation_summary([_fev_predictions(predictor, train_df)], model_name="")[eval_metric]

    assert np.isclose(ag_score, fev_score)


def _reference_seasonal_error_per_item(arrays, seasonality, aggregate_fn):
    """Simple for-loop reference implementation for testing."""
    result = []
    for arr in arrays:
        if len(arr) <= seasonality:
            result.append(float("nan"))
        else:
            diffs = arr[seasonality:] - arr[:-seasonality]
            errors = aggregate_fn(diffs)
            mean_error = np.nanmean(errors)
            result.append(mean_error)
    return np.array(result, dtype="float64")


def _arrays_to_flat(arrays):
    """Helper to convert list of 1D arrays to flat [total_T, 1] + lengths [N]."""
    lengths = np.array([len(a) for a in arrays], dtype=np.int64)
    flat = np.concatenate(arrays).astype(np.float64).reshape(-1, 1) if arrays else np.empty((0, 1), dtype=np.float64)
    return flat, lengths


@pytest.mark.parametrize("aggregate_fn", [np.abs, np.square])
def test_seasonal_error_per_item(aggregate_fn):
    """Test vectorized impl against reference with mixed edge cases."""
    arrays = [
        np.array([1.0]),  # too short
        np.array([1.0, 2.0, np.nan, 4.0, 5.0]),  # has NaN
        np.array([np.nan, np.nan, np.nan, np.nan]),  # all NaN
        np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),  # normal
        np.array([1.0, 2.0]),  # too short
        np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]),  # longer series
    ]
    seasonality = 2

    flat, lengths = _arrays_to_flat(arrays)
    result = _seasonal_error_per_item(
        y_past=flat, y_past_lengths=lengths, seasonality=seasonality, aggregate_fn=aggregate_fn
    )[:, 0]
    expected = _reference_seasonal_error_per_item(arrays, seasonality, aggregate_fn)

    np.testing.assert_allclose(result, expected)


def test_seasonal_error_per_item_empty():
    """Test with empty input."""
    flat, lengths = _arrays_to_flat([])
    result = _seasonal_error_per_item(y_past=flat, y_past_lengths=lengths, seasonality=2, aggregate_fn=np.abs)
    assert result.size == 0
    assert result.dtype == np.float64


@pytest.mark.parametrize("metric_name", ["MQL", "WQL", "SQL"])
def test_when_per_quantile_scores_then_overall_equals_mean_of_per_level(model_setup, metric_name):
    task, train_df, _, predictor = model_setup
    task.eval_metric = metric_name

    summary = task.evaluation_summary([_fev_predictions(predictor, train_df)], model_name="", per_quantile_scores=True)

    per_level = [summary[f"{metric_name}[{q}]"] for q in task.quantile_levels]
    assert np.isclose(summary[metric_name], np.mean(per_level))


@pytest.mark.parametrize("metric_name", ["MQL", "WQL", "SQL"])
def test_when_per_quantile_scores_disabled_then_no_per_level_keys(model_setup, metric_name):
    task, train_df, _, predictor = model_setup
    task.eval_metric = metric_name

    summary = task.evaluation_summary([_fev_predictions(predictor, train_df)], model_name="")

    assert metric_name in summary
    assert not any(key.startswith(f"{metric_name}[") for key in summary)


def test_when_per_quantile_scores_then_non_quantile_metrics_have_no_breakdown(model_setup):
    task, train_df, _, predictor = model_setup
    task.eval_metric = "MASE"
    task.extra_metrics = ["MAE", "SQL"]

    summary = task.evaluation_summary([_fev_predictions(predictor, train_df)], model_name="", per_quantile_scores=True)

    # Quantile metric is broken down per level
    assert all(f"SQL[{q}]" in summary for q in task.quantile_levels)
    # Non-quantile metrics emit only their overall score
    assert not any(key.startswith("MAE[") or key.startswith("MASE[") for key in summary)

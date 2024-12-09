import datasets
import numpy as np
import pandas as pd
import pytest
from autogluon.timeseries import TimeSeriesPredictor

import fev
from fev.metrics import AVAILABLE_METRICS


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
    train, future = task.get_input_data()
    test = task.get_test_data()

    train_df = _to_pandas(train).rename(columns={"id": "item_id"})
    test_df = _to_pandas(test).rename(columns={"id": "item_id"})

    predictor = TimeSeriesPredictor(
        prediction_length=task.horizon,
        eval_metric_seasonal_period=task.seasonality,
        quantile_levels=task.quantile_levels,
        path=tmp_path_factory.mktemp("predictor"),
    ).fit(train_df, hyperparameters={"SeasonalNaive": {}}, verbosity=0)

    return task, train_df, test_df, predictor


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

    ag_predictions = predictor.predict(train_df).rename(columns={"mean": "predictions"})
    fev_predictions = []
    for _, pred in ag_predictions.groupby("item_id", as_index=False):
        fev_predictions.append(pred.to_dict("list"))

    fev_score = task.compute_metrics(fev_predictions)[eval_metric]

    assert np.isclose(ag_score, fev_score)

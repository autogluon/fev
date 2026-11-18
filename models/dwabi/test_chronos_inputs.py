"""Check the D-wabi cold-start fallback against the pinned Chronos input parser."""

import numpy as np
import pytest
from _dwabi_fev.chronos_missing import prepare_fev_inputs
from chronos.chronos2 import dataset


def test_cold_start_is_scoped_to_the_provider_and_keeps_numeric_plans():
    parse = dataset.validate_and_prepare_single_dict_input
    categories = np.array(["a", "b", "a"])
    plans = np.array([8.0, 13.0])
    raw = {
        "target": np.full(3, np.nan),
        "past_covariates": {"category": categories, "plan": np.arange(3.0)},
        "future_covariates": {"category": np.array(["b", "a"]), "plan": plans},
    }
    observed = raw | {"target": np.array([0.0, np.nan, 2.0])}
    multivariate = raw | {"target": np.full((2, 3), np.nan)}

    def parse_rows(inputs, prediction_length):
        return [parse(row, idx, prediction_length) for idx, row in enumerate(inputs)]

    evaluate = prepare_fev_inputs(parse_rows)
    cold, warm, multiple = evaluate(inputs=[raw, observed, multivariate], prediction_length=2)
    assert cold["n_future_covariates"] == 2
    np.testing.assert_array_equal(cold["future_covariates"][1], [np.nan, np.nan])
    np.testing.assert_array_equal(cold["future_covariates"][2], plans)
    np.testing.assert_array_equal(raw["future_covariates"]["category"], ["b", "a"])
    for actual, item in [(warm, observed), (multiple, multivariate)]:
        reference = parse(item, 0, 2)
        for field in ("context", "future_covariates"):
            np.testing.assert_array_equal(actual[field], reference[field])

    assert dataset.validate_and_prepare_single_dict_input is parse
    with pytest.raises(ValueError, match="0 sample"):
        dataset.validate_and_prepare_single_dict_input(raw, 0, 2)

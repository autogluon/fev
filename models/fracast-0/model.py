"""Fracast-0 adapter for temporary FEV-Bench experiments."""
from __future__ import annotations

import datasets
import numpy as np
import torch

import fev


class Fracast0Model(fev.ForecastingModel):
    model_name = "fracast-0"
    trained_on_datasets = [
        "ETT_15T",
        "ETT_1D",
        "ETT_1H",
        "ETT_1W",
        "LOOP_SEATTLE_1D",
        "LOOP_SEATTLE_1H",
        "LOOP_SEATTLE_5T",
        "M_DENSE_1D",
        "M_DENSE_1H",
        "SZ_TAXI_15T",
        "SZ_TAXI_1H",
        "australian_tourism",
        "bizitobs_l2c_1H",
        "bizitobs_l2c_5T",
        "boomlet_1062",
        "boomlet_1209",
        "boomlet_1225",
        "boomlet_1230",
        "boomlet_1282",
        "boomlet_1487",
        "boomlet_1631",
        "boomlet_1676",
        "boomlet_1855",
        "boomlet_1975",
        "boomlet_2187",
        "boomlet_285",
        "boomlet_619",
        "boomlet_772",
        "boomlet_963",
        "ecdc_ili",
        "entsoe_15T",
        "entsoe_1H",
        "entsoe_30T",
        "epf_be",
        "epf_de",
        "epf_fr",
        "epf_np",
        "epf_pjm",
        "ercot_1D",
        "ercot_1H",
        "ercot_1M",
        "ercot_1W",
        "favorita_stores_1D",
        "favorita_stores_1M",
        "favorita_stores_1W",
        "favorita_transactions_1D",
        "favorita_transactions_1M",
        "favorita_transactions_1W",
        "fred_md_2025",
        "fred_qd_2025",
        "gvar",
        "hermes",
        "hierarchical_sales_1D",
        "hierarchical_sales_1W",
        "hospital",
        "hospital_admissions_1D",
        "hospital_admissions_1W",
        "jena_weather_10T",
        "jena_weather_1D",
        "jena_weather_1H",
        "kdd_cup_2022_10T",
        "kdd_cup_2022_1D",
        "kdd_cup_2022_30T",
        "m5_1D",
        "m5_1M",
        "m5_1W",
        "proenfo_gfc12",
        "proenfo_gfc14",
        "proenfo_gfc17",
        "redset_15T",
        "redset_1H",
        "redset_5T",
        "restaurant",
        "rohlik_orders_1D",
        "rohlik_orders_1W",
        "rohlik_sales_1D",
        "rohlik_sales_1W",
        "rossmann_1D",
        "rossmann_1W",
        "solar_1D",
        "solar_1W",
        "solar_with_weather_15T",
        "solar_with_weather_1H",
        "uci_air_quality_1D",
        "uci_air_quality_1H",
        "uk_covid_nation_1D",
        "uk_covid_nation_1W",
        "uk_covid_utla_1D",
        "uk_covid_utla_1W",
        "us_consumption_1M",
        "us_consumption_1Q",
        "us_consumption_1Y",
        "walmart",
        "world_co2_emissions",
        "world_life_expectancy",
        "world_tourism",
    ]

    def __init__(
        self,
        model_id: str = "ztxtech/fracast-0",
        weights: str = "w8",
        batch_size: int = 512,
        device: str = "cpu",
    ):
        super().__init__()
        from fracast import FracastModel

        self.model = FracastModel.from_pretrained(
            model_id, weights=weights, device=device
        )
        self.batch_size = int(batch_size)

    @staticmethod
    def _targets(window: fev.EvaluationWindow) -> list[list[float]]:
        past_data, _ = fev.convert_input_data(
            window, adapter="datasets", as_univariate=True
        )
        return past_data.with_format("numpy").cast_column(
            "target", datasets.Sequence(datasets.Value("float32"))
        )["target"]

    def _fit_predict(self, task: fev.Task) -> list[datasets.DatasetDict]:
        predictions_per_window = []
        for window in task.iter_windows():
            targets = self._targets(window)
            normalized_contexts = []
            for values in targets:
                context = np.asarray(values, dtype=np.float32)
                context = context[-self.model.context_length :]
                context[~np.isfinite(context)] = np.nan
                normalized_contexts.append(context)
            width = max(len(context) for context in normalized_contexts)
            contexts = np.full(
                (len(normalized_contexts), width), np.nan, dtype=np.float32
            )
            for row_index, context in enumerate(normalized_contexts):
                contexts[row_index, -len(context) :] = context
            task_fallback_rows = 0
            with self._record_inference_time():
                finite_counts = np.isfinite(contexts).sum(axis=1)
                fallback_indices = np.flatnonzero(finite_counts < 8)
                valid_indices = np.flatnonzero(finite_counts >= 8)
                chunks = [
                    self.model.forecast_batch(
                        contexts[valid_indices[start : start + self.batch_size]],
                        horizon=task.horizon,
                    )
                    for start in range(0, len(valid_indices), self.batch_size)
                ]
                valid_forecasts = np.concatenate(chunks, axis=0) if chunks else np.empty(
                    (0, task.horizon, len(task.quantile_levels)), dtype=np.float32
                )
                forecasts = np.empty(
                    (len(contexts), task.horizon, len(task.quantile_levels)),
                    dtype=np.float32,
                )
                forecasts[valid_indices] = valid_forecasts
                task_fallback_rows = len(fallback_indices)
                if task_fallback_rows:
                    for row_index in fallback_indices:
                        finite_columns = np.flatnonzero(np.isfinite(contexts[row_index]))
                        fallback_value = (
                            contexts[row_index, finite_columns[-1]]
                            if len(finite_columns)
                            else np.float32(0.0)
                        )
                        forecasts[row_index] = fallback_value
                    print(
                        f"fracast-0 fallback rows on {task.task_name}: "
                        f"{task_fallback_rows}",
                        flush=True,
                    )
            predictions_dict = {
                "predictions": forecasts[:, :, np.argmin(np.abs(np.array(self.model.quantiles) - 0.5))],
            }
            if task.quantile_levels != self.model.quantiles:
                raise ValueError(
                    f"Fracast-0 quantiles {self.model.quantiles} do not match task "
                    f"quantiles {task.quantile_levels}"
                )
            for level_index, level in enumerate(task.quantile_levels):
                predictions_dict[str(level)] = forecasts[:, :, level_index]
            predictions_per_window.append(
                fev.utils.combine_univariate_predictions_to_multivariate(
                    datasets.Dataset.from_dict(predictions_dict),
                    target_columns=task.target_columns,
                )
            )
        return predictions_per_window

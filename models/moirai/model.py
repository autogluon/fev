import logging
import warnings

import datasets
import numpy as np

import fev


class MoiraiModel(fev.ForecastingModel):
    """Moirai v2 from https://huggingface.co/Salesforce/moirai-2.0-R-small."""

    model_name = "moirai"
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
        "bizitobs_l2c_1H",
        "bizitobs_l2c_5T",
        "favorita_transactions_1D",
        "fred_md_2025",
        "hierarchical_sales_1D",
        "hierarchical_sales_1W",
        "hospital",
        "jena_weather_10T",
        "jena_weather_1D",
        "jena_weather_1H",
        "kdd_cup_2022_10T",
        "m5_1D",
        "proenfo_gfc12",
        "proenfo_gfc14",
        "proenfo_gfc17",
        "restaurant",
    ]

    def __init__(
        self,
        model_path: str = "Salesforce/moirai-2.0-R-small",
        batch_size: int = 128,
        max_context_length: int = 4000,
        device: str = "auto",
        ignore_covariates: bool = True,
        as_univariate: bool = True,
    ):
        super().__init__()
        self.model_path = model_path
        self.batch_size = batch_size
        self.max_context_length = max_context_length
        self.device = device
        self.ignore_covariates = ignore_covariates
        self.as_univariate = as_univariate

    def _fit_predict(self, task: fev.Task) -> list[datasets.DatasetDict]:
        import torch
        from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

        model_path = fev.utils.maybe_cache_from_s3(self.model_path)
        module = Moirai2Module.from_pretrained(model_path)

        gts_dataset, _ = fev.convert_input_data(task.get_window(0), "gluonts", as_univariate=self.as_univariate)
        if self.as_univariate:
            target_columns = ["target"]
        else:
            target_columns = task.target_columns

        if self.ignore_covariates:
            feat_dynamic_real_dim = 0
            past_feat_dynamic_real_dim = 0
        else:
            feat_dynamic_real_dim = gts_dataset.num_feat_dynamic_real
            past_feat_dynamic_real_dim = gts_dataset.num_past_feat_dynamic_real

        model = Moirai2Forecast(
            module=module,
            prediction_length=task.horizon,
            context_length=self.max_context_length,
            target_dim=len(target_columns),
            feat_dynamic_real_dim=feat_dynamic_real_dim,
            past_feat_dynamic_real_dim=past_feat_dynamic_real_dim,
        )

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        predictor = model.create_predictor(batch_size=self.batch_size, device=self.device)

        predictions_per_window = []
        for window in task.iter_windows():
            _, prediction_dataset = fev.convert_input_data(window, adapter="gluonts", as_univariate=self.as_univariate)
            with self._record_inference_time():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    forecasts = list(predictor.predict(prediction_dataset))

            predictions = self._postprocess_forecasts(
                forecasts, target_columns=target_columns, quantile_levels=task.quantile_levels
            )
            predictions_per_window.append(
                fev.utils.combine_univariate_predictions_to_multivariate(
                    predictions, target_columns=task.target_columns
                )
            )
        return predictions_per_window

    @staticmethod
    def _postprocess_forecasts(
        forecasts: list, quantile_levels: list[float], target_columns: list[str]
    ) -> datasets.DatasetDict:
        gts_logger = logging.getLogger("gluonts")
        gts_logger.setLevel(100)

        forecast_keys = ["predictions"] + [str(q) for q in quantile_levels]
        predictions_per_key = {key: [] for key in forecast_keys}
        num_dims = len(target_columns)
        for f in forecasts:
            predictions_per_key["predictions"].append(f.mean.reshape([-1, num_dims]))
            for q in quantile_levels:
                predictions_per_key[str(q)].append(f.quantile(q).reshape([-1, num_dims]))
        for key in forecast_keys:
            predictions_per_key[key] = np.stack(predictions_per_key[key])

        predictions = {}
        for i, target in enumerate(target_columns):
            predictions[target] = datasets.Dataset.from_dict(
                {key: predictions_per_key[key][..., i] for key in forecast_keys}
            )
        return datasets.DatasetDict(predictions)

import datasets
import numpy as np

import fev


class IAmTimeModel(fev.ForecastingModel):
	"""iAmTime forecasting model from https://github.com/anishsaha12/iAmTime."""

	model_name = "iamtime"
	trained_on_datasets = []

	def __init__(
		self,
		model_path: str = "anishsaha/iAmTime-base-forecast",
		device: str = "cuda",
		torch_dtype: str = "bfloat16",
		batch_size: int = 64,
		use_exogenous: bool = True,
		use_past_dynamic: bool = True,
		use_known_dynamic: bool = True,
	):
		super().__init__()
		if batch_size < 1:
			raise ValueError("batch_size must be positive")
		self.model_path = model_path
		self.device = device
		self.torch_dtype = torch_dtype
		self.batch_size = batch_size
		self.use_exogenous = use_exogenous
		self.use_past_dynamic = use_past_dynamic
		self.use_known_dynamic = use_known_dynamic
		self._pipeline = None

	def _load_pipeline(self):
		import torch
		from iAmTime.pipeline import iAmTimePipeline

		if self._pipeline is None:
			device = self.device
			if device == "auto":
				device = "cuda" if torch.cuda.is_available() else "cpu"

			dtype = self.torch_dtype
			if dtype == "auto":
				dtype = (
					torch.bfloat16
					if device == "cuda" and torch.cuda.is_bf16_supported()
					else torch.float32
				)
			elif isinstance(dtype, str):
				dtype_names = {"bfloat16": torch.bfloat16, "float32": torch.float32}
				if dtype not in dtype_names:
					raise ValueError(f"Unsupported torch_dtype: {dtype}")
				dtype = dtype_names[dtype]

			self._pipeline = iAmTimePipeline.from_pretrained(
				self.model_path,
				device_map=device,
				torch_dtype=dtype,
			)
			self._pipeline.model.eval()
		return self._pipeline

	def _fit_predict(self, task: fev.Task) -> list[datasets.DatasetDict]:
		import iAmTime.icl_data_factory as IclDataFactory

		pipeline = self._load_pipeline()
		predictions_per_window = []

		for window in task.iter_windows():
			past_data, future_data = fev.convert_input_data(
				window, adapter="datasets", as_univariate=True
			)

			quantile_levels = task.quantile_levels.copy()
			if 0.5 not in quantile_levels:
				quantile_levels.append(0.5)

			quantile_batches = []
			mean_batches = []
			for start in range(0, len(past_data), self.batch_size):
				end = min(start + self.batch_size, len(past_data))
				batch_past = past_data.select(range(start, end))
				batch_future = future_data.select(range(start, end))
				batch_inputs, _, _, _ = IclDataFactory.convert_past_future_datasets_to_icl_input(
					past_data=batch_past,
					future_data=batch_future,
					target_columns=["target"],
					past_dynamic_columns=window.past_dynamic_columns,
					known_dynamic_columns=window.known_dynamic_columns,
					prediction_length=window.horizon,
					context_length=pipeline.model.iamtime_config.context_length,
					min_examples=0,
					max_examples=0,
					min_past_for_examples=0,
					use_exogenous=self.use_exogenous,
					use_past_dynamic=self.use_past_dynamic,
					use_known_dynamic=self.use_known_dynamic,
				)
				with self._record_inference_time():
					quantiles, mean = pipeline.predict_quantiles(
						icl_inputs=batch_inputs,
						prediction_length=window.horizon,
						quantile_levels=quantile_levels,
					)
					quantile_batches.append(quantiles.detach().cpu().numpy())
					mean_batches.append(mean.detach().cpu().numpy())

			quantiles_np = np.concatenate(quantile_batches, axis=0)
			mean_np = np.concatenate(mean_batches, axis=0)
			if task.eval_metric in ["MSE", "RMSE", "RMSSE"]:
				point_forecast = mean_np
			else:
				point_forecast = quantiles_np[:, :, quantile_levels.index(0.5)]

			prediction_columns = {"predictions": point_forecast}
			for index, level in enumerate(quantile_levels):
				if level in task.quantile_levels:
					prediction_columns[str(level)] = quantiles_np[:, :, index]

			predictions = fev.utils.combine_univariate_predictions_to_multivariate(
				datasets.Dataset.from_dict(prediction_columns),
				target_columns=window.target_columns,
			)
			predictions_per_window.append(predictions)

		return predictions_per_window

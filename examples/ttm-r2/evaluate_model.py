import time

import datasets
import numpy as np
import pandas as pd
import torch
from gluonts.transform import LastValueImputation
from tqdm.auto import tqdm
from tsfm_public import TinyTimeMixerForPrediction

import fev

datasets.disable_progress_bars()


class TinyTimeMixerPipeline:
    def __init__(self, model):
        self.model = model
        self.max_context_length = model.config.context_length
        self.max_prediction_length = self.model.config.prediction_length

    def _left_pad_and_stack_1D(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        max_len = max(self.max_context_length, *(len(c) for c in tensors))
        padded = []
        for c in tensors:
            assert isinstance(c, torch.Tensor)
            assert c.ndim == 1
            padding = torch.full(size=(max_len - len(c),), fill_value=torch.nan, device=c.device)
            padded.append(torch.concat((padding, c), dim=-1))
        return torch.stack(padded)

    def _prepare_and_validate_context(self, context: torch.Tensor | list[torch.Tensor]):
        if isinstance(context, list):
            context = self._left_pad_and_stack_1D(context)
        assert isinstance(context, torch.Tensor)
        if context.ndim == 1:
            context = context.unsqueeze(0)
        assert context.ndim == 2

        if context.size(-1) > self.max_context_length:
            context = context[..., -self.max_context_length :]

        # Add channel dimension
        context = context.unsqueeze(dim=-1)

        return context

    def predict(
        self,
        context: torch.Tensor | list[torch.Tensor],
        prediction_length: int,
    ):
        assert prediction_length <= self.max_prediction_length, (
            f"Only prediction lengths up to {self.max_prediction_length} are supported"
        )

        context = self._prepare_and_validate_context(context).float()
        nan_mask = torch.isnan(context)
        context[nan_mask] = 0.0

        with torch.no_grad():
            output = self.model(
                past_values=context.to(self.model.device),
                past_observed_mask=~nan_mask.to(self.model.device),
            ).prediction_outputs.cpu()  # (batch, prediction_length, 1)

        # truncate predictions
        return output.squeeze(dim=-1)[:, :prediction_length]

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        model = TinyTimeMixerForPrediction.from_pretrained(*args, **kwargs)
        model.eval()
        return cls(model=model)


def batchify(lst: list, batch_size: int = 32):
    """Convert list into batches of desired size."""
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]


def predict_with_model(
    task: fev.Task,
    model_name: str = "ibm-granite/granite-timeseries-ttm-r2",
    device: str = "cuda",
    batch_size: int = 256,
) -> tuple[datasets.Dataset, float, dict]:
    past_data, _ = task.get_input_data(trust_remote_code=True)
    target = past_data.with_format("numpy").cast_column(
        task.target_column, datasets.Sequence(datasets.Value("float32"))
    )[task.target_column]

    imputation = LastValueImputation()
    inputs = [imputation(t) for t in target]

    pipeline = TinyTimeMixerPipeline.from_pretrained(model_name, device_map=device)

    forecast_per_batch = []
    start_time = time.monotonic()
    for batch in tqdm(batchify(inputs, batch_size=batch_size), total=len(inputs) // batch_size):
        forecast_per_batch.append(
            pipeline.predict([torch.tensor(x) for x in batch], prediction_length=task.horizon).cpu().numpy()
        )
    inference_time = time.monotonic() - start_time

    forecast = np.concatenate(forecast_per_batch, axis=0)  # [num_items, horizon]
    predictions_dict = {"predictions": forecast}
    # Probabilistic forecasting not supported, so we repeat the point forecast for each quantile
    if task.quantile_levels is not None:
        for q in task.quantile_levels:
            predictions_dict[str(q)] = forecast
    predictions = datasets.Dataset.from_dict(predictions_dict)

    extra_info = {
        "model_config": {
            "context_length": pipeline.max_context_length,
            "model_name": model_name,
            "batch_size": batch_size,
            "device": device,
        }
    }
    return predictions, inference_time, extra_info


if __name__ == "__main__":
    model_name = "ibm-granite/granite-timeseries-ttm-r2"
    num_tasks = 2  # replace with `num_tasks = None` to run on all tasks

    benchmark = fev.Benchmark.from_yaml(
        "https://raw.githubusercontent.com/autogluon/fev/refs/heads/main/benchmarks/chronos_zeroshot/tasks.yaml"
    )
    summaries = []
    for task in benchmark.tasks[:num_tasks]:
        predictions, inference_time, extra_info = predict_with_model(task, model_name=model_name)
        evaluation_summary = task.evaluation_summary(
            predictions,
            model_name=model_name,
            inference_time_s=inference_time,
            extra_info=extra_info,
        )
        print(evaluation_summary)
        summaries.append(evaluation_summary)

    # Show and save the results
    summary_df = pd.DataFrame(summaries)
    print(summary_df)
    summary_df.to_csv("ttm-r2.csv", index=False)

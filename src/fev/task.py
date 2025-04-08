import copy
import dataclasses
import pprint
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Literal

import datasets
import numpy as np
import pandas as pd
import pydantic

from . import utils
from .__about__ import __version__ as FEV_VERSION
from .constants import DEFAULT_NUM_PROC, FUTURE, PREDICTIONS, TEST, TRAIN
from .metrics import AVAILABLE_METRICS, QUANTILE_METRICS

MULTIPLE_TARGET_COLUMNS_ALL = "__ALL__"


@pydantic.dataclasses.dataclass(config={"extra": "forbid"})
class _TaskBase:
    """Base class defining the attributes of a time series forecasting task."""

    dataset_path: str | None = None
    dataset_config: str | None = None
    # Forecast horizon parameters
    horizon: int = 1
    cutoff: int | str | None = None
    lead_time: int = 1
    min_ts_length: int | None = None
    max_context_length: int | None = None
    # Evaluation parameters
    seasonality: int = 1
    eval_metric: str = "MASE"
    extra_metrics: list[str] = dataclasses.field(default_factory=list)
    quantile_levels: list[float] | None = None
    # Feature information
    id_column: str = "id"
    timestamp_column: str = "timestamp"
    target_column: str = "target"
    multiple_target_columns: list[str] | Literal["__ALL__"] | None = None
    past_dynamic_columns: list[str] = dataclasses.field(default_factory=list)
    excluded_columns: list[str] = dataclasses.field(default_factory=list)


@pydantic.dataclasses.dataclass
class Task(_TaskBase):
    """
    A univariate time series forecasting task.

    This object is responsible for

    - loading the data
    - generating a train/test split
    - evaluating the predictions accuracy

    A single `Task` object corresponds to a single train-test split of the data. This means that, for example, to
    perform evaluation on `N` rolling windows, it is necessary to create `N` separate `Task` objects (one for each
    window).

    Parameters
    ----------
    dataset_path : str
        Path to the time series dataset stored locally, on S3, or on Hugging Face Hub. See the Examples section below
        for information on how to load datasets from different sources.
    dataset_config : str | None, default None
        Name of the configuration used when loading datasets from Hugging Face Hub. If `dataset_config` is provided,
        the datasets will be loaded from HF Hub. If `dataset_config=None`, the dataset will be loaded from a local or
        S3 path.
    horizon : int, default 1
        Length of the forecast horizon (in time steps).
    cutoff : int | str | None, default -horizon
        Position in the series where that divides the observed data and the forecast horizon. Defaults to `-horizon`.
        Cutoff logic is similar to pandas indexing:

        - If `cutoff` is an integer, then `y[:cutoff]` is the observed data and `y[cutoff]` is the first value in the forecast horizon. Positive or negative integer values are allowed.
        - If `cutoff` is a datetime-like string, then `y[cutoff]` is the last observation in the observed data.

        If some time series are too short for the chosen cutoff (i.e., there are no observations before the cutoff or
        there are fewer than `horizon` observations after the cutoff), an exception will be raised when loading the
        data.
    lead_time : int, default 1
        Number of time steps between the end of observed data and the start of the forecast horizon.
    min_ts_length : int | None, default horizon + 1
        Time series with length less than `min_ts_length` will be removed from the dataset. Defaults to `horizon + 1`.
    max_context_length : int | None, default None
        If provided, the past time series will be shortened to at most this many observations.
    seasonality : int, default 1
        Seasonal period of the dataset (e.g., 24 for hourly data, 12 for monthly data). This parameter is used when
        computing metrics like Mean Absolute Scaled Error.
    eval_metric : str, default 'MASE'
        Evaluation metric used for ultimate evaluation on the test set.
    extra_metrics : list[str], default None
        Additional metrics to be included in the results.
    quantile_levels : list[float] | None, default None
        Quantiles that must be predicted. List of floats between 0 and 1 (for example, [0.1, 0.5, 0.9]).
    id_column : str, default 'id'
        Name of the column with the unique identifier of each time series.
    timestamp_column : str, default 'timestamp'
        Name of the column with the timestamps of the observations.
    target_column : str, default 'target'
        Name of the column that must be predicted.
    multiple_target_columns : list[str] | Literal["__ALL__"] | None, default None
        If provided, a separate univariate time series will be created from each of the multiple_target_columns fields.

        If set to `"__ALL__"`, then a separate univariate instance will be created from each column of type `Sequence`.

        For example, if `multiple_target_columns = ["X", "Y"]` then the raw multivariate time series
        `{"id": "A", "timestamp": [...], "X": [...], "Y": [...]}` will be split into two univariate time series
        `{"id": "A_X", "timestamp": [...], "target": [...]}` and `{"id": "A_Y", "timestamp": [...], "target": [...]}`.
    past_dynamic_columns : list[str], default None
        Names of columns that are known only in the past. These will be available in the past data, but not in the
        future or test data.
    excluded_columns : list[str], default None
        Names of columns that are removed from the dataset during preprocessing.

    Examples
    --------
    Dataset stored on the Hugging Face Hub

    >>> Task(dataset_path="autogluon/chronos_datasets", dataset_config="m4_hourly", ...)

    Dataset stored as a parquet file (local or S3)

    >>> Task(dataset_path="s3://my-bucket/m4_hourly/data.parquet", ...)

    Dataset consisting of multiple parquet files (local or S3)

    >>> Task(dataset_path="s3://my-bucket/m4_hourly/*.parquet", ...)
    """

    def __post_init__(self):
        # TODO: Add support for lead_time > 1
        if self.lead_time > 1:
            raise ValueError("lead_time > 1 is currently not supported")

        if self.dataset_path is None:
            raise ValueError("`dataset_path` cannot be `None` when creating a `Task`")

        if self.cutoff is None:
            self.cutoff = -self.horizon
        elif isinstance(self.cutoff, str):
            self.cutoff = pd.Timestamp(self.cutoff).isoformat()
        else:
            self.cutoff = int(self.cutoff)
            if self.cutoff < 0 and self.cutoff > -self.horizon:
                raise ValueError("Negative `cutoff` must be less than or equal to `-horizon`")

        self.eval_metric = self.eval_metric.upper()
        self.extra_metrics = [m.upper() for m in self.extra_metrics]
        for metric in [self.eval_metric] + self.extra_metrics:
            if metric not in AVAILABLE_METRICS:
                raise ValueError(
                    f"Evaluation metric '{metric}' is not available. Available metrics: {sorted(AVAILABLE_METRICS)}"
                )
            if metric in QUANTILE_METRICS and self.quantile_levels is None:
                raise ValueError(f"Please set quantile_levels when using a quantile metric '{metric}'")

        if self.quantile_levels is not None:
            assert all(0 < q < 1 for q in self.quantile_levels), "All quantile_levels must satisfy 0 < q < 1"
            self.quantile_levels = sorted(self.quantile_levels)
        if self.min_ts_length is None:
            self.min_ts_length = self.horizon + 1
        if self.max_context_length is not None:
            if self.max_context_length < 1:
                raise ValueError("If provided, `max_context_length` must satisfy >= 1")
        self._dataset_dict: datasets.DatasetDict | None = None
        # Attributes computed after the dataset is loaded
        self._freq: str | None = None
        self._dataset_fingerprint: str | None = None
        self._dynamic_columns: list[str] | None = None
        self._static_columns: list[str] | None = None

    def to_dict(self) -> dict:
        """Convert task definition to a dictionary."""
        return dataclasses.asdict(self)

    @property
    def dataset_name(self) -> str:
        """Human-readable name of the dataset obtained from dataset path and config name."""
        if self.dataset_config is None:
            # File dataset -> name of the parent directory
            return Path(self.dataset_path).parent.name
        else:
            # HF Hub dataset -> name of the repo + config name
            return self.dataset_path.split("/")[-1] + "_" + self.dataset_config

    def get_input_data(
        self,
        num_proc: int = DEFAULT_NUM_PROC,
        storage_options: dict | None = None,
        trust_remote_code: bool | None = None,
    ) -> tuple[datasets.Dataset, datasets.Dataset]:
        """Get data that is available as input to the model. This includes all past data and known future data.

        Parameters
        ----------
        num_proc : int
            Number of CPU cores used to parallelize dataset operations.
        storage_options : dict | None, default None
            Storage options passed to the `datasets.load_dataset` method.

            For example, to load data from a private S3 bucket only accessible via the AWS profile `my-aws-profile`
            defined in `~/.aws/config`, set `storage_options={"profile": "my-aws-profile"}`.
        trust_remote_code : bool | None, default None
            Whether to trust custom builders.

        Returns
        -------
        past_data : datasets.Dataset
            Past data available to the model. This includes static features and past values of target & the dynamic
            features.

            Contains columns corresponding to `task.id_column`, `task.timestamp_column`, `task.target_column`,
            `task.static_columns` and `task.dynamic_columns`.
        future_data : datasets.Dataset
            Known future data for making the predictions. This includes future values of the dynamic features.

            Contains columns corresponding to `task.id_column`, `task.timestamp_column`, `task.static_columns` and
            `task.dynamic_columns` (except those in `task.past_dynamic_columns`).
        """
        if self._dataset_dict is None:
            self._prepare_dataset_dict(
                num_proc=num_proc, storage_options=storage_options, trust_remote_code=trust_remote_code
            )
        return self._dataset_dict[TRAIN], self._dataset_dict[FUTURE]

    def get_test_data(
        self,
        num_proc: int = DEFAULT_NUM_PROC,
        storage_options: dict | None = None,
        trust_remote_code: bool | None = None,
    ) -> datasets.Dataset:
        """Get the future data that must be predicted.

        This data is only used for evaluating forecast accuracy and should not be passed to the model.
        """
        if self._dataset_dict is None:
            self._prepare_dataset_dict(
                num_proc=num_proc, storage_options=storage_options, trust_remote_code=trust_remote_code
            )
        return self._dataset_dict[TEST]

    @property
    def freq(self) -> str:
        """Pandas string corresponding to the frequency of the time series in the dataset.

        See [pandas documentation](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#cutoff-aliases)
        for the list of possible values.
        """
        if self._freq is None:
            raise ValueError("Please load dataset first with `task.get_input_data()`")
        return self._freq

    @property
    def dynamic_columns(self) -> list[str]:
        if self._dynamic_columns is None:
            raise ValueError("Please load dataset first with `task.get_input_data()`")
        return self._dynamic_columns

    @property
    def known_dynamic_columns(self) -> list[str]:
        if self._dynamic_columns is None:
            raise ValueError("Please load dataset first with `task.get_input_data()`")
        return sorted(set(self.dynamic_columns) - set(self.past_dynamic_columns))

    @property
    def static_columns(self) -> list[str]:
        if self._static_columns is None:
            raise ValueError("Please load dataset first with `task.get_input_data()`")
        return self._static_columns

    def _load_dataset(
        self,
        storage_options: dict | None = None,
        trust_remote_code: bool | None = None,
        num_proc: int = DEFAULT_NUM_PROC,
    ) -> datasets.Dataset:
        """Load the raw dataset from the provided path."""
        if self.dataset_config is not None:
            # Load dataset from HF Hub
            path = self.dataset_path
            name = self.dataset_config
            data_files = None
        else:
            # Load dataset from a local or remote file
            dataset_format = Path(self.dataset_path).suffix.lstrip(".")
            allowed_formats = ["parquet", "arrow"]
            if dataset_format not in allowed_formats:
                raise ValueError(f"When loading dataset from file, path must end in one of {allowed_formats}.")
            path = dataset_format
            name = None
            data_files = self.dataset_path

        if storage_options is None:
            storage_options = {}

        load_dataset_kwargs = dict(
            path=path,
            name=name,
            data_files=data_files,
            split=TRAIN,
            storage_options=copy.deepcopy(storage_options),
            trust_remote_code=trust_remote_code,
        )
        try:
            ds = datasets.load_dataset(
                **load_dataset_kwargs,
                # PatchedDownloadConfig fixes https://github.com/huggingface/datasets/issues/6598
                download_config=utils.PatchedDownloadConfig(storage_options=copy.deepcopy(storage_options)),
            )
        except Exception:
            raise RuntimeError(
                "Failed to load the dataset when calling `datasets.load_dataset` with arguments\n"
                f"{pprint.pformat(load_dataset_kwargs)}"
            )
        ds.set_format("numpy")

        required_columns = self.past_dynamic_columns + self.excluded_columns
        if self.multiple_target_columns is None:
            required_columns += [self.target_column]
        elif self.multiple_target_columns == MULTIPLE_TARGET_COLUMNS_ALL:
            pass
        else:
            required_columns += self.multiple_target_columns

        utils.validate_time_series_dataset(
            ds,
            id_column=self.id_column,
            timestamp_column=self.timestamp_column,
            required_columns=required_columns,
            num_proc=num_proc,
        )

        ds = ds.remove_columns(self.excluded_columns)
        return ds

    def _filter_short_series(
        self,
        dataset: datasets.Dataset,
        num_proc: int,
    ) -> datasets.Dataset:
        """Remove records from the datasets that have length lower than self.min_ts_length."""
        return dataset.filter(
            _is_record_long_enough,
            fn_kwargs=dict(timestamp_column=self.timestamp_column, min_ts_length=self.min_ts_length),
            num_proc=num_proc,
            desc="Filtering short time series",
        )

    def _past_future_test_split(
        self,
        dataset: datasets.Dataset,
        columns_to_slice: list[str],
        num_proc: int,
    ) -> datasets.DatasetDict:
        past_data = dataset.map(
            _select_past,
            fn_kwargs=dict(
                columns_to_slice=columns_to_slice,
                timestamp_column=self.timestamp_column,
                cutoff=self.cutoff,
                max_context_length=self.max_context_length,
            ),
            num_proc=num_proc,
            desc="Selecting past data",
        )

        future_data = dataset.map(
            _select_future,
            fn_kwargs=dict(
                columns_to_slice=columns_to_slice,
                timestamp_column=self.timestamp_column,
                cutoff=self.cutoff,
                horizon=self.horizon,
            ),
            num_proc=num_proc,
            desc="Selecting future data",
        )
        future_known = future_data.remove_columns([self.target_column] + self.past_dynamic_columns)
        test = future_data.select_columns([self.id_column, self.timestamp_column, self.target_column])
        return datasets.DatasetDict({TRAIN: past_data, FUTURE: future_known, TEST: test})

    def _prepare_dataset_dict(
        self,
        num_proc: int,
        storage_options: dict | None = None,
        trust_remote_code: bool | None = None,
    ) -> None:
        """Load dataset and split it into past, future and test parts."""
        ds = self._load_dataset(
            storage_options=storage_options,
            trust_remote_code=trust_remote_code,
            num_proc=num_proc,
        )
        if self.multiple_target_columns is not None:
            if self.multiple_target_columns == MULTIPLE_TARGET_COLUMNS_ALL:
                multiple_target_columns = [
                    col
                    for col, feat in ds.features.items()
                    if isinstance(feat, datasets.Sequence) and col != self.timestamp_column
                ]
            else:
                multiple_target_columns = self.multiple_target_columns
            ds = ds.map(
                _expand_target_columns,
                batched=True,
                fn_kwargs=dict(
                    id_column=self.id_column,
                    target_column=self.target_column,
                    multiple_target_columns=multiple_target_columns,
                ),
                remove_columns=multiple_target_columns,
                num_proc=num_proc,
            )

        # Ensure that IDs are sorted alphabetically for consistent ordering
        if ds.features[self.id_column].dtype != "string":
            ds = ds.cast_column(self.id_column, datasets.Value("string"))
        ds = ds.sort(self.id_column)
        self._freq = pd.infer_freq(ds[0][self.timestamp_column])
        if self._freq is None:
            raise ValueError("Dataset contains irregular timestamps")
        self._dataset_fingerprint = utils.generate_fingerprint(ds)

        self._dynamic_columns, self._static_columns = utils.infer_column_types(
            ds,
            id_column=self.id_column,
            timestamp_column=self.timestamp_column,
        )
        self._dynamic_columns.remove(self.target_column)
        columns_to_slice = self.dynamic_columns + [self.target_column, self.timestamp_column]
        num_proc = min(num_proc, len(ds))
        ds = self._filter_short_series(ds, num_proc=num_proc)
        self._dataset_dict = self._past_future_test_split(ds, columns_to_slice=columns_to_slice, num_proc=num_proc)

    @property
    def dataset_info(self) -> dict:
        return {
            "id_column": self.id_column,
            "timestamp_column": self.timestamp_column,
            "target_column": self.target_column,
            "static_columns": self.static_columns,
            "dynamic_columns": self.dynamic_columns,
            "known_dynamic_columns": self.known_dynamic_columns,
            "past_dynamic_columns": self.past_dynamic_columns,
        }

    @property
    def predictions_schema(self) -> datasets.Features:
        """Describes the format that the predictions must follow.

        Forecast must always include the key `"predictions"` corresponding to the point forecast.

        Moreover, if `quantile_levels` were specified when creating the `Task`, then predictions must contain a key for
        each of the predicted quantiles (e.g., if `quantile_levels = [0.1, 0.9]`, then keys `"0.1"` and `"0.9"` must be
        included in the forecast).
        """
        predictions_length = self.horizon + self.lead_time - 1
        predictions_schema = {
            PREDICTIONS: datasets.Sequence(datasets.Value("float64"), length=predictions_length),
        }
        if self.quantile_levels is not None:
            for q in sorted(self.quantile_levels):
                predictions_schema[str(q)] = datasets.Sequence(datasets.Value("float64"), length=predictions_length)
        return datasets.Features(predictions_schema)

    def compute_metrics(self, predictions: datasets.Dataset | list[dict]) -> dict[str, float]:
        test_data = self.get_test_data().with_format("numpy")
        past_data = self._dataset_dict[TRAIN].with_format("numpy")
        predictions = self._clean_and_validate_predictions(predictions)

        if len(predictions) != len(test_data):
            raise ValueError(
                f"Length of predictions ({len(predictions)}) must match the length of test data ({len(test_data)})"
            )

        test_scores = {}
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            for eval_metric in sorted(set([self.eval_metric] + self.extra_metrics)):
                metric = AVAILABLE_METRICS[eval_metric]()
                test_scores[eval_metric] = float(
                    metric.compute(
                        test_data=test_data,
                        predictions=predictions,
                        past_data=past_data,
                        seasonality=self.seasonality,
                        quantile_levels=self.quantile_levels,
                        target_column=self.target_column,
                    )
                )
        return test_scores

    def _clean_and_validate_predictions(self, predictions: datasets.Dataset | list[dict]) -> datasets.Dataset:
        """Convert predictions to the format needed for computing the metrics."""
        if isinstance(predictions, Iterable):
            try:
                predictions = datasets.Dataset.from_list(list(predictions))
            except Exception:
                raise ValueError(
                    "`datasets.Dataset.from_list(predictions)` failed. Please convert predictions to `datasets.Dataset` format."
                )
        if not isinstance(predictions, datasets.Dataset):
            raise ValueError(f"predictions must be of type `datasets.Dataset` (received {type(predictions)})")
        predictions = predictions.cast(self.predictions_schema).with_format("numpy")

        for col in predictions.column_names:
            nan_row_idx, _ = np.where(~np.isfinite(np.array(predictions[col])))
            if len(nan_row_idx) > 0:
                raise ValueError(
                    "Predictions contain NaN or Inf values. "
                    f"First invalid value encountered in column {col} for item {nan_row_idx[0]}:\n"
                    f"{predictions[int(nan_row_idx[0])]}"
                )
        return predictions

    def evaluation_summary(
        self,
        predictions: datasets.Dataset | list[dict],
        model_name: str,
        training_time_s: float | None = None,
        inference_time_s: float | None = None,
        trained_on_this_dataset: bool = False,
        extra_info: dict | None = None,
    ) -> dict:
        """Get a summary of the model performance for the given forecasting task.

        Parameters
        ----------
        predictions : list[dict] | datasets.Dataset
            Predictions generated by the model. The predictions must follow the format described in `task.predictions_schema`.
        model_name : str
            Name of the model that generated the predictions.
        training_time_s : float | None
            Training time of the model for this task (in seconds).
        inference_time_s : float | None
            Total inference time to generate all predictions (in seconds).
        trained_on_this_dataset : bool
            Was the model trained on the dataset associated with this task? Set to False if the model is used in
            zero-shot mode.
        extra_info : dict | None
            Optional dictionary with additional information that will be appended to the evaluation summary.

        Returns
        -------
        summary : dict
            Dictionary that summarizes the model performance on this task. Includes following keys:

            - `model_name` - name of the model
            - `test_error` - value of the `task.eval_metric` achieved by the `predictions` on the test set (lower is better)
            - `dataset_name` - human-readable name of the dataset
            - all `Task` attributes obtained via `task.to_dict()`.
            - values of `task.extra_metrics` achieved by the `predictions` on the test
            - `dataset_fingerprint` - fingerprint of the dataset generated by the HF `datasets` library
            - `trained_on_this_dataset` - whether the model was trained on the dataset used in the task
            - `fev_version` - version of the `fev` package used to obtain the summary
        """
        summary = {
            "model_name": model_name,
            "dataset_name": self.dataset_name,
        }
        summary.update(self.to_dict())
        metric_scores = self.compute_metrics(predictions)
        summary.update(
            {
                "test_error": float(metric_scores[self.eval_metric]),
                "training_time_s": training_time_s,
                "inference_time_s": inference_time_s,
                "dataset_fingerprint": self._dataset_fingerprint,
                "trained_on_this_dataset": trained_on_this_dataset,
                "fev_version": FEV_VERSION,
            }
        )
        summary.update(metric_scores)
        if extra_info is not None:
            summary.update(extra_info)
        return summary


@pydantic.dataclasses.dataclass
class TaskGenerator(_TaskBase):
    """
    Can generate one or multiple `Task` objects based on the task configuration.

    Supports the same keyword arguments as `Task`, in addition to the following arguments for defining multiple
    variants of the same task.

    Parameters
    ----------
    variants : list[dict] | None
        List, where each entry corresponds to a variant of the base task. See *Examples* for usage details.
        If `variants` are provided together with the rolling evaluation arguments (`num_rolling_windows`,
        `rolling_step_size` or `initial_cutoff`), an exception will be raised.
    num_rolling_windows : int | None
        Number of rolling evaluation windows to generate from the base task.
    initial_cutoff : int | str | None
        Cutoff for the first rolling window. Can be a negative integer (e.g., `-48`) or a timestamp-like string
        (e.g., `"2024-02-01"). See also documentation for `cutoff` argument to `Task`.
        Defaults to `-num_rolling_windows * rolling_step_size`.
    rolling_step_size : int | str | None
        Step size between consecutive rolling evaluation windows.
        If `initial_cutoff` is an integer, `rolling_step_size` must be a positive integer.
        If `initial_cutoff` is a timestamp-like string, `rolling_step_size` must be pandas-compatible offset string
        (e.g., `D` for daily, `15min` for quarter-hourly).
        Defaults to `horizon`.

    Examples
    --------
    To define a single task, simply define the respective attributes:

    >>> task_config = TaskGenerator(
    ...     dataset_path="my_dataset",
    ...     dataset_config="my_config",
    ...     horizon=12,
    ... )
    >>> print(task_config.generate_tasks())
    [Task(dataset_path='my_dataset', dataset_config="my_config", horizon=12, ...)]

    To create multiple variants of the same task, you can use the `variants` keyword:

    >>> task_config = TaskGenerator(
    ...     dataset_path="my_dataset",
    ...     dataset_config="my_config",
    ...     variants=[
    ...         {"horizon": 12},
    ...         {"horizon": 24},
    ...     ]
    ... )
    >>> print(task_config.generate_tasks())
    [Task(dataset_path='my_dataset', dataset_config="my_config", horizon=12, ...),
     Task(dataset_path='my_dataset', dataset_config="my_config", horizon=24, ...)]

    Alternatively, you can configure rolling evaluation using the keywords `num_rolling_windows`, `rolling_step_size`
    and `initial_cutoff`.

    Using integer-based cutoffs:

    >>> task_config = TaskGenerator(
    ...     dataset_path="my_dataset",
    ...     dataset_config="my_config",
    ...     horizon=24,
    ...     num_rolling_windows=3,
    ...     initial_cutoff=-96,
    ...     rolling_step_size=None,  # defaults to `horizon`
    ... )
    >>> print(task_config.generate_tasks())
    [Task(dataset_path='my_dataset', dataset_config="my_config", horizon=24, cutoff=-96, ...),
     Task(dataset_path='my_dataset', dataset_config="my_config", horizon=24, cutoff=-72, ...),
     Task(dataset_path='my_dataset', dataset_config="my_config", horizon=24, cutoff=-48, ...)]

    Using timestamp-based cutoffs:

    >>> task_config = TaskGenerator(
    ...     dataset_path="my_dataset",
    ...     dataset_config="my_config",
    ...     horizon=24,
    ...     num_rolling_windows=3,
    ...     initial_cutoff="2024-01-05",
    ...     rolling_step_size="12h",  # required
    ... )
    >>> print(task_config.generate_tasks())
    [Task(dataset_path='my_dataset', dataset_config="my_config", horizon=24, cutoff="2024-01-05T00:00:00", ...),
     Task(dataset_path='my_dataset', dataset_config="my_config", horizon=24, cutoff="2024-01-05T12:00:00", ...),
     Task(dataset_path='my_dataset', dataset_config="my_config", horizon=24, cutoff="2024-01-06T00:00:00", ...)]

    """

    variants: list[dict[str, Any]] | None = None
    num_rolling_windows: int | None = None
    initial_cutoff: int | str | None = None
    rolling_step_size: int | str | None = None

    def __post_init__(self):
        if self.variants is not None:
            assert self.num_rolling_windows is None, "`num_rolling_windows` must be `None` if `variants` is provided"
            assert self.initial_cutoff is None, "`num_rolling_windows` must be `None` if `variants` is provided"
            assert self.rolling_step_size is None, "`rolling_step_size` must be `None` if `variants` is provided"
        elif self.num_rolling_windows is not None:
            assert self.variants is None, "`variants` must be `None` if `num_rolling_windows` is provided"
            assert self.num_rolling_windows >= 1, "If provided, `num_rolling_windows` must satisfy >= 1"
            if self.rolling_step_size is None:
                self.rolling_step_size = self.horizon
            if self.initial_cutoff is None:
                self.initial_cutoff = -self.num_rolling_windows * self.horizon

            if isinstance(self.initial_cutoff, int):
                if not isinstance(self.rolling_step_size, int):
                    raise ValueError("`rolling_step_size` must be an int if `initial_cutoff` is an int")
                assert self.initial_cutoff <= -1
                assert self.rolling_step_size >= 1
            else:
                if not isinstance(self.rolling_step_size, str):
                    raise ValueError("`rolling_step_size` must be a string if `initial_cutoff` is a string")
                self.initial_cutoff = pd.Timestamp(self.initial_cutoff).isoformat()
                offset = pd.tseries.frequencies.to_offset(self.rolling_step_size)
                assert offset.n >= 1, "If `rolling_step_size` is a string, it must correspond to a positive timedelta"
                self.rolling_step_size = offset.freqstr

    def generate_tasks(self) -> list[Task]:
        tasks = []
        excluded_keys = ["variants", "num_rolling_windows", "rolling_step_size", "initial_cutoff"]
        base_task_data = {k: v for k, v in self.__dict__.items() if k not in excluded_keys}

        if self.variants:
            for variant in self.variants:
                task_data = base_task_data.copy()
                task_data.update(variant)
                tasks.append(Task(**task_data))
        elif self.num_rolling_windows:
            for window_idx in range(self.num_rolling_windows):
                task_data = base_task_data.copy()
                if isinstance(self.initial_cutoff, int):
                    cutoff = self.initial_cutoff + window_idx * self.rolling_step_size
                else:
                    cutoff = pd.Timestamp(self.initial_cutoff)
                    # We don't add the offset for window_idx=0 to avoid applying an "anchored" offset
                    # (e.g. `Timestamp("2020-01-01") + i * to_offset("ME")` returns "2020-01-31" for i=0 and i=1)
                    if window_idx != 0:
                        cutoff += window_idx * pd.tseries.frequencies.to_offset(self.rolling_step_size)
                    cutoff = cutoff.isoformat()
                task_data["cutoff"] = cutoff
                tasks.append(Task(**task_data))
        else:
            tasks.append(Task(**base_task_data))
        return tasks


# These methods are stored outside of classes to ensure that HF datasets caching logic recognizes them
def _select_past(
    record: dict,
    columns_to_slice: list[str],
    timestamp_column: str,
    cutoff: int | str,
    max_context_length: int | None,
) -> dict:
    """Select values up to cutoff in the columns_to_slice."""
    if isinstance(cutoff, str):
        selection = record[timestamp_column] <= np.datetime64(cutoff)
    else:
        selection = slice(None, cutoff)
    processed_record = {}
    for name, value in record.items():
        if name in columns_to_slice:
            value = value[selection]
            if len(value) < 1:
                raise ValueError(f"Sequences too short to create the train set with {cutoff=} for record {record}")
            if max_context_length is not None:
                value = value[-max_context_length:]
        processed_record[name] = value
    return processed_record


def _select_future(
    record: dict,
    columns_to_slice: list[str],
    timestamp_column: str,
    horizon: int,
    cutoff: int | str,
) -> dict:
    """Select horizon values after the cutoff in the columns_to_slice."""
    if isinstance(cutoff, str):
        selection = record[timestamp_column] > np.datetime64(cutoff)
    else:
        selection = slice(cutoff, None)
    processed_record = {}
    for name, value in record.items():
        if name in columns_to_slice:
            value = value[selection][:horizon]
            if len(value) < horizon:
                raise ValueError(
                    f"Sequences too short to create the test set with {cutoff=} and {horizon=} for record {record}"
                )
        processed_record[name] = value
    return processed_record


def _is_record_long_enough(record: dict, timestamp_column: str, min_ts_length: int) -> bool:
    """Return True if time series length is >= min_ts_length."""
    return len(record[timestamp_column]) >= min_ts_length


def _expand_target_columns(
    batch: dict, id_column: str, target_column: str, multiple_target_columns: list[str]
) -> dict:
    """Create a separate record for each column listed in multiple_target_columns.

    It is required to set batched=True when using method in `dataset.map`.
    """
    expanded_batch = defaultdict(list)
    batch_size = len(batch[id_column])
    for i in range(batch_size):
        for target_col in multiple_target_columns:
            for key in batch.keys():
                if key not in multiple_target_columns:
                    value = batch[key][i]
                    if key == id_column:
                        value = value + "_" + target_col
                    expanded_batch[key].append(value)
            expanded_batch[target_column].append(batch[target_col][i])
    return dict(expanded_batch)

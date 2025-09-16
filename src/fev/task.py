import copy
import dataclasses
import logging
import pprint
import warnings
from pathlib import Path
from typing import Any, Iterable, Literal

import datasets
import numpy as np
import pandas as pd
import pydantic
from pydantic_core import ArgsKwargs

from . import utils
from .__about__ import __version__ as FEV_VERSION
from .constants import DEFAULT_NUM_PROC, DEPRECATED_TASK_FIELDS, FUTURE, PREDICTIONS, TEST, TRAIN
from .metrics import Metric, get_metric

# from .metrics import AVAILABLE_METRICS, QUANTILE_METRICS

ALL_AVAILABLE_COLUMNS: Literal["__ALL__"] = "__ALL__"

logger = logging.getLogger("fev")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# Disable progress bar at `import fev` to spawning too many progress bars
datasets.disable_progress_bar()


@dataclasses.dataclass
class EvaluationWindow:
    """
    A single evaluation window on which the forecast accuracy is measured.

    Corresponds to a single train/test split of the time series data at the provided `cutoff`.

    You should never manually create `EvaluationWindow` objects. Instead, use [`Task.iter_windows()`][fev.Task.iter_windows]
    or [`Task.get_window()`][fev.Task.get_window] to obtain the evaluation windows corresponding to the task.
    """

    full_dataset: datasets.Dataset = dataclasses.field(repr=False)
    cutoff: int | str
    horizon: int
    min_context_length: int | None
    max_context_length: int | None
    # Dataset info
    id_column: str
    timestamp_column: str
    target_columns: list[str]
    known_dynamic_columns: list[str]
    past_dynamic_columns: list[str]
    static_columns: list[str]

    def __post_init__(self):
        self._dataset_dict: datasets.DatasetDict | None = None

    def get_input_data(self, num_proc: int = DEFAULT_NUM_PROC) -> tuple[datasets.Dataset, datasets.Dataset]:
        """Get data available to the model at prediction time for this evaluation window.

        To convert the input data to a different format, use [`fev.convert_input_data`][fev.convert_input_data].

        Parameters
        ----------
        num_proc : int, default DEFAULT_NUM_PROC
            Number of processes to use when splitting the dataset.

        Returns
        -------
        past_data : datasets.Dataset
            Historical observations up to the cutoff point.
            Contains: id, timestamps, target values, static covariates, and all dynamic covariates.

            Columns corresponding to `id_column`, `timestamp_column`, `target_columns`, `static_columns`,
            `past_dynamic_columns`, `known_dynamic_columns`.
        future_data : datasets.Dataset
            Known future information for the forecast horizon.

            Columns corresponding to `id_column`, `timestamp_column`, `static_columns`, `known_dynamic_columns`.
        """
        if self._dataset_dict is None:
            self._dataset_dict = self._prepare_dataset_dict(num_proc=num_proc)
        return self._dataset_dict[TRAIN], self._dataset_dict[FUTURE]

    def get_ground_truth(self, num_proc: int = DEFAULT_NUM_PROC) -> datasets.Dataset:
        """Get ground truth future test data.

        **This data should never be provided to the model!**

        This is a convenience method that exists for debugging and additional evaluation.

        Parameters
        ----------
        num_proc : int, default DEFAULT_NUM_PROC
            Number of processes to use when splitting the dataset.
        """
        if self._dataset_dict is None:
            self._dataset_dict = self._prepare_dataset_dict(num_proc=num_proc)
        return self._dataset_dict[TEST]

    def compute_metrics(
        self,
        predictions: datasets.DatasetDict,
        metrics: list[Metric],
        seasonality: int,
        quantile_levels: list[float],
    ) -> dict[str, float]:
        """Compute accuracy metrics on the predictions made for this window.

        To compute metrics on your predictions, use [`Task.evaluation_summary`][fev.Task.evaluation_summary] instead.

        This is a convenience method that exists for debugging and additional evaluation.
        """
        test_data = self.get_ground_truth().with_format("numpy")
        past_data = self.get_input_data()[0].with_format("numpy")

        for target_column, predictions_for_column in predictions.items():
            if len(predictions_for_column) != len(test_data):
                raise ValueError(
                    f"Length of predictions for column {target_column} ({len(predictions)}) must "
                    f"match the length of test data ({len(test_data)})"
                )

        test_scores: dict[str, float] = {}
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            for metric in metrics:
                scores = []
                for col in self.target_columns:
                    scores.append(
                        metric.compute(
                            test_data=test_data,
                            predictions=predictions[col],
                            past_data=past_data,
                            seasonality=seasonality,
                            quantile_levels=quantile_levels,
                            target_column=col,
                        )
                    )
                test_scores[metric.name] = float(np.mean(scores))
        return test_scores

    def _filter_short_series(
        self,
        dataset: datasets.Dataset,
        num_proc: int,
    ) -> datasets.Dataset:
        """Remove records from the dataset that are too short for the given task configuration.

        Filters out time series if they have either fewer than `min_context_length` observations before `cutoff`, or
        fewer than `horizon` observations after `cutoff`.
        """
        num_items_before = len(dataset)
        filtered_dataset = dataset.filter(
            _has_enough_past_and_future_observations,
            fn_kwargs=dict(
                timestamp_column=self.timestamp_column,
                horizon=self.horizon,
                cutoff=self.cutoff,
                min_context_length=self.min_context_length,
            ),
            num_proc=min(num_proc, len(dataset)),
            desc="Filtering short time series",
        )
        num_items_after = len(filtered_dataset)
        if num_items_after < num_items_before:
            logger.info(
                f"Dropped {num_items_before - num_items_after} out of {num_items_before} time series "
                f"because they had fewer than min_context_length ({self.min_context_length}) "
                f"observations before cutoff ({self.cutoff}) "
                f"or fewer than horizon ({self.horizon}) "
                f"observations after cutoff."
            )
        if len(filtered_dataset) == 0:
            raise ValueError(
                "All time series in the dataset are too short for the chosen cutoff, horizon and min_context_length"
            )
        return filtered_dataset

    def _prepare_dataset_dict(self, num_proc: int = DEFAULT_NUM_PROC) -> datasets.DatasetDict:
        dataset = self.full_dataset.select_columns(
            [self.id_column, self.timestamp_column]
            + self.target_columns
            + self.known_dynamic_columns
            + self.past_dynamic_columns
            + self.static_columns
        )
        dataset = self._filter_short_series(dataset, num_proc=num_proc)
        columns_to_slice = [col for col, feat in dataset.features.items() if isinstance(feat, datasets.Sequence)]
        past_data = dataset.map(
            _select_past,
            fn_kwargs=dict(
                columns_to_slice=columns_to_slice,
                timestamp_column=self.timestamp_column,
                cutoff=self.cutoff,
                max_context_length=self.max_context_length,
            ),
            num_proc=min(num_proc, len(dataset)),
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
            num_proc=min(num_proc, len(dataset)),
            desc="Selecting future data",
        )
        future_known = future_data.remove_columns(self.target_columns + self.past_dynamic_columns)
        test = future_data.select_columns([self.id_column, self.timestamp_column] + self.target_columns)
        return datasets.DatasetDict({TRAIN: past_data, FUTURE: future_known, TEST: test})


@pydantic.dataclasses.dataclass(config={"extra": "forbid"})
class Task:
    """A univariate or multivariate time series forecasting task.

    A `Task` stores all information uniquely identifying the task, such as path to the dataset, forecast horizon,
    evaluation metric and names of the target & covariate columns.

    This object handles dataset loading, train/test splitting, and prediction evaluation for time series forecasting tasks.

    A single `Task` consists of one or more [`EvaluationWindow`][fev.task.EvaluationWindow] objects that can be
    accessed using [`iter_windows()`][fev.Task.iter_windows] or [`get_window()`][fev.Task.get_window] methods.
    After making predictions for each evaluation window, you can evaluate their accuracy using [`evaluation_summary()`][fev.Task.evaluation_summary].

    Typical workflow:
    ```python
    task = fev.Task(dataset_path="...", num_windows=3, horizon=24)

    predictions_per_window = []
    for window in task.iter_windows():
        past_data, future_data = window.get_input_data()
        predictions = model.predict(past_data, future_data)
        predictions_per_window.append(predictions)

    summary = task.evaluation_summary(predictions_per_window, model_name="my_model")
    ```

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
    num_windows : int, default 1
        Number of rolling evaluation windows included in the task.
    initial_cutoff : int | str | None, default None
        Starting position for the first evaluation window that separates past from future data.

        Can be specified as:

        - *Integer*: Index position using Python indexing. `y[:initial_cutoff]` becomes the past/training data,
            and `y[initial_cutoff:initial_cutoff+horizon]` becomes the first forecast horizon to predict.
            Negative values are interpreted as steps from the end of the series.
        - *Timestamp string*: Date or datetime (e.g., `"2024-02-01"`). Data up to and including this timestamp becomes
            past data, and the next `horizon` observations become the first forecast horizon.

        If `None`, defaults to `-horizon - (num_windows - 1) * window_step_size`.

        **Note**: Time series that are too short for any evaluation window (i.e., have fewer than `min_context_length`
        observations before a cutoff or fewer than `horizon` observations after a cutoff) will be filtered out during
        data loading.
    window_step_size : int | str | None, default horizon
        Step size between consecutive evaluation windows. Must be an integer if `initial_cutoff` is an integer.
        Can be an integer or pandas offset string (e.g., `'D'`, `'15min'`) if `initial_cutoff` is a timestamp.
        Defaults to `horizon`.
    min_context_length : int, default 1
        Time series with fewer than `min_context_length` observations before a cutoff will be ignored during evaluation.
    max_context_length : int | None, default None
        If provided, the past time series will be shortened to at most this many observations.
    seasonality : int, default 1
        Seasonal period of the dataset (e.g., 24 for hourly data, 12 for monthly data). This parameter is used when
        computing metrics like Mean Absolute Scaled Error (MASE).
    eval_metric : str | dict[str, Any], default 'MASE'
        Evaluation metric used for ultimate evaluation on the test set. Can be specified as either a single string
        with the metric's name or a dictionary containing a "name" key and extra hyperparameters for the metric.
        For example, MASE can also be specified as `{"name": "MASE", "epsilon": 0.0001}` to prevent zero
        denominators when scaling errors.
    extra_metrics : list[str] | list[dict[str, Any]], default []
        Additional metrics to be included in the results. Can be specified as a list of strings with the metric's
        name or a list of dictionaries. See documentation for `eval_metric` for more details.
    quantile_levels : list[float], default []
        Quantiles that must be predicted. List of floats between 0 and 1 (for example, `[0.1, 0.5, 0.9]`).
    id_column : str, default 'id'
        Name of the column with the unique identifier of each time series.
        This column will be casted to `string` dtype and the dataset will be sorted according to it.
    timestamp_column : str, default 'timestamp'
        Name of the column with the timestamps of the observations.
    target : str | list[str], default 'target'
        Name of the column that must be predicted. If a string is provided, a univariate forecasting task is created.
        If a list of strings is provided, a multivariate forecasting task is created.
    generate_univariate_targets_from : list[str] | Literal["__ALL__"] | None, default None
        If provided, a separate univariate time series will be created from each of the
        `generate_univariate_targets_from` columns. Only valid for univariate tasks.

        If set to `"__ALL__"`, then a separate univariate instance will be created from each column of type `Sequence`.

        For example, if `generate_univariate_targets_from = ["X", "Y"]` then the raw multivariate time series
        `{"id": "A", "timestamp": [...], "X": [...], "Y": [...]}` will be split into two univariate time series
        `{"id": "A_X", "timestamp": [...], "target": [...]}` and `{"id": "A_Y", "timestamp": [...], "target": [...]}`.
    past_dynamic_columns : list[str], default []
        Names of covariate columns that are known only in the past. These will be available in the past data, but not
        in the future data. An error will be raised if these columns are missing from the dataset.
    known_dynamic_columns : list[str], default []
        Names of covariate columns that are known in both past and future. These will be available in past data
        and future data. An error will be raised if these columns are missing from the dataset.
    static_columns : list[str], default []
        Names of columns containing static covariates that don't change over time. An error will be raised if these
        columns are missing from the dataset.
    task_name : str | None, default None
        Human-readable name for the task. Defaults to `dataset_config` for datasets stored on HF hub, and to the
        name of 2 parent directories for local or S3-based datasets.

        This field is only here for convenience and is not used for any validation when computing the results.

    Examples
    --------
    Dataset stored on the Hugging Face Hub

    >>> Task(dataset_path="autogluon/chronos_datasets", dataset_config="m4_hourly", ...)

    Dataset stored as a parquet file (local or S3)

    >>> Task(dataset_path="s3://my-bucket/m4_hourly/data.parquet", ...)

    Dataset consisting of multiple parquet files (local or S3)

    >>> Task(dataset_path="s3://my-bucket/m4_hourly/*.parquet", ...)
    """

    dataset_path: str
    dataset_config: str | None = None
    # Forecast horizon parameters
    horizon: int = 1
    num_windows: int = 1
    initial_cutoff: int | str | None = None
    window_step_size: int | str | None = None
    min_context_length: int = 1
    max_context_length: int | None = None
    # Evaluation parameters
    seasonality: int = 1
    eval_metric: str | dict[str, Any] = "MASE"
    extra_metrics: list[str | dict[str, Any]] = dataclasses.field(default_factory=list)
    quantile_levels: list[float] = dataclasses.field(default_factory=list)
    # Feature information
    id_column: str = "id"
    timestamp_column: str = "timestamp"
    target: str | list[str] = "target"
    generate_univariate_targets_from: list[str] | Literal["__ALL__"] | None = None
    known_dynamic_columns: list[str] = dataclasses.field(default_factory=list)
    past_dynamic_columns: list[str] = dataclasses.field(default_factory=list)
    static_columns: list[str] = dataclasses.field(default_factory=list)
    task_name: str | None = None

    def __post_init__(self):
        if self.task_name is None:
            if self.dataset_config is not None:
                # HF Hub dataset -> name of dataset_config
                self.task_name = self.dataset_config
            else:
                # File dataset -> names of up to 2 parent directories
                # e.g. /home/foo/bar/data.parquet -> foo/bar
                self.task_name = "/".join(Path(self.dataset_path).parts[-3:-1])

        assert self.num_windows >= 1, "`num_windows` must satisfy >= 1"
        if self.window_step_size is None:
            self.window_step_size = self.horizon
        if isinstance(self.window_step_size, int):
            assert self.window_step_size >= 1, "`window_step_size` must satisfy >= 1"
        else:
            offset = pd.tseries.frequencies.to_offset(self.window_step_size)
            assert offset.n >= 1, "If `window_step_size` is a string, it must correspond to a positive timedelta"
            self.window_step_size = offset.freqstr

        if self.initial_cutoff is None:
            assert isinstance(self.window_step_size, int), (
                "If `initial_cutoff` is None, `window_step_size` must be an int"
            )
            self.initial_cutoff = -self.horizon - (self.num_windows - 1) * self.window_step_size

        if isinstance(self.initial_cutoff, int):
            if not isinstance(self.window_step_size, int):
                raise ValueError("`window_step_size` must be an int if `initial_cutoff` is an int")
            assert self.window_step_size >= 1
            max_allowed_cutoff = -self.horizon - (self.num_windows - 1) * self.window_step_size
            if self.initial_cutoff < 0 and self.initial_cutoff > max_allowed_cutoff:
                raise ValueError(
                    "Negative `initial_cutoff` must be <= `-horizon - (num_windows - 1) * window_step_size`"
                )
        else:
            self.initial_cutoff = pd.Timestamp(self.initial_cutoff).isoformat()

        assert all(0 < q < 1 for q in self.quantile_levels), "All quantile_levels must satisfy 0 < q < 1"
        self.quantile_levels = sorted(self.quantile_levels)

        metrics = [get_metric(m) for m in [self.eval_metric] + self.extra_metrics]

        metric_names = [m.name for m in metrics]
        duplicate_metric_names = {x for x in metric_names if metric_names.count(x) > 1}
        if duplicate_metric_names:
            raise ValueError(
                f"Duplicate metric names found: {duplicate_metric_names}. Please configure "
                "only one instance for each metric."
            )

        if len(self.quantile_levels) == 0:
            for m in metrics:
                if m.needs_quantiles:
                    raise ValueError(f"Please provide quantile_levels when using a quantile metric '{m.name}'")

        if self.min_context_length < 1:
            raise ValueError("`min_context_length` must satisfy >= 1")
        if self.max_context_length is not None:
            if self.max_context_length < 1:
                raise ValueError("If provided, `max_context_length` must satisfy >= 1")

        if isinstance(self.target, list):
            if len(self.target) < 1:
                raise ValueError("For multivariate tasks `target` must contain at least one entry")
            # Ensure that column names are sorted alphabetically so that univariate adapters return sorted data
            self.target = sorted(self.target)

        # Ensure that column names are sorted alphabetically for deterministic task comparison
        self.known_dynamic_columns = sorted(self.known_dynamic_columns)
        self.past_dynamic_columns = sorted(self.past_dynamic_columns)
        self.static_columns = sorted(self.static_columns)

        if self.generate_univariate_targets_from is not None and self.is_multivariate:
            raise ValueError(
                "`generate_univariate_targets_from` cannot be used for multivariate tasks (when `target` is a list)"
            )

        # Attributes computed after the dataset is loaded
        self._full_dataset: datasets.Dataset | None = None
        self._freq: str | None = None
        self._dataset_fingerprint: str | None = None

    @property
    def cutoffs(self) -> list[int] | list[str]:
        """Cutoffs corresponding to each `EvaluationWindow` in the task.

        Computed based on `num_windows`, `initial_cutoff` and `window_step_size` attributes of the task.
        """
        if isinstance(self.initial_cutoff, int):
            assert isinstance(self.window_step_size, int)
            return [self.initial_cutoff + window_idx * self.window_step_size for window_idx in range(self.num_windows)]
        else:
            assert isinstance(self.initial_cutoff, str)
            if isinstance(self.window_step_size, str):
                offset = pd.tseries.frequencies.to_offset(self.window_step_size)
            else:
                assert isinstance(self.window_step_size, int)
                offset = pd.tseries.frequencies.to_offset(self.freq) * self.window_step_size

            cutoffs = []
            for window_idx in range(self.num_windows):
                cutoff = pd.Timestamp(self.initial_cutoff)
                # We don't add the offset for window_idx=0 to avoid applying an "anchored" offset
                # (e.g. `Timestamp("2020-01-01") + i * to_offset("ME")` returns "2020-01-31" for i=0 and i=1)
                if window_idx != 0:
                    cutoff += window_idx * offset
                cutoffs.append(cutoff.isoformat())
            return cutoffs

    def to_dict(self) -> dict:
        """Convert task definition to a dictionary."""
        return dataclasses.asdict(self)

    def load_full_dataset(
        self,
        storage_options: dict | None = None,
        trust_remote_code: bool | None = None,
        num_proc: int = DEFAULT_NUM_PROC,
    ) -> datasets.Dataset:
        """Load the full raw dataset with preprocessing applied.

        This method validates the data, loads and preprocesses the dataset according to the task configuration,
        including generating univariate targets if `generate_univariate_targets_from` is provided.

        **Note:** This method is only provided for information and debugging purposes. For model evaluation, use
        [`iter_windows()`][fev.Task.iter_windows] instead to get properly split train/test data.

        Parameters
        ----------
        storage_options : dict, optional
            Passed to `datasets.load_dataset()` for accessing remote datasets (e.g., S3 credentials).
        trust_remote_code : bool, optional
            Passed to `datasets.load_dataset()` for trusting remote code from Hugging Face Hub.
        num_proc : int, default DEFAULT_NUM_PROC
            Number of processes to use for dataset preprocessing.

        Returns
        -------
        datasets.Dataset
            The preprocessed dataset with all time series.
        """
        if self._full_dataset is None:
            self._full_dataset = self._load_dataset(
                storage_options=storage_options, trust_remote_code=trust_remote_code, num_proc=num_proc
            )
        return self._full_dataset

    def iter_windows(
        self,
        storage_options: dict | None = None,
        trust_remote_code: bool | None = None,
        num_proc: int = DEFAULT_NUM_PROC,
    ) -> Iterable[EvaluationWindow]:
        """Iterate over the rolling evaluation windows in the task.

        Each window contains train/test splits at different cutoff points for time series
        cross-validation. Use this method for model evaluation and benchmarking.

        Parameters
        ----------
        storage_options : dict, optional
            Passed to `datasets.load_dataset()` for accessing remote datasets (e.g., S3 credentials).
        trust_remote_code : bool, optional
            Passed to `datasets.load_dataset()` for trusting remote code from Hugging Face Hub.
        num_proc : int, default DEFAULT_NUM_PROC
            Number of processes to use for dataset preprocessing.

        Yields
        ------
        EvaluationWindow
            A single evaluation window at a specific cutoff containing the data needed to make and evaluate forecasts.

        Examples
        --------
        >>> for window in task.iter_windows():
        ...     past_data, future_data = window.get_input_data()
        ...     # Make predictions using past_data and future_data
        """
        for window_idx in range(self.num_windows):
            yield self.get_window(
                window_idx, storage_options=storage_options, trust_remote_code=trust_remote_code, num_proc=num_proc
            )

    def get_window(
        self,
        window_idx: int,
        storage_options: dict | None = None,
        trust_remote_code: bool | None = None,
        num_proc: int = DEFAULT_NUM_PROC,
    ) -> EvaluationWindow:
        """Get a single evaluation window by index.

        Parameters
        ----------
        window_idx : int
            Index of the evaluation window in [0, 1, ..., num_windows - 1].
        storage_options : dict, optional
            Passed to `datasets.load_dataset()` for accessing remote datasets (e.g., S3 credentials).
        trust_remote_code : bool, optional
            Passed to `datasets.load_dataset()` for trusting remote code from Hugging Face Hub.
        num_proc : int, default DEFAULT_NUM_PROC
            Number of processes to use for dataset preprocessing.

        Returns
        -------
        EvaluationWindow
            A single evaluation window at a specific cutoff containing the data needed to make and evaluate forecasts.
        """
        full_dataset = self.load_full_dataset(
            storage_options=storage_options, trust_remote_code=trust_remote_code, num_proc=num_proc
        )
        if window_idx >= self.num_windows:
            raise ValueError(f"Window index {window_idx} is out of range (num_windows={self.num_windows})")
        return EvaluationWindow(
            full_dataset=full_dataset,
            cutoff=self.cutoffs[window_idx],
            horizon=self.horizon,
            min_context_length=self.min_context_length,
            max_context_length=self.max_context_length,
            id_column=self.id_column,
            timestamp_column=self.timestamp_column,
            target_columns=self.target_columns,
            known_dynamic_columns=self.known_dynamic_columns,
            past_dynamic_columns=self.past_dynamic_columns,
            static_columns=self.static_columns,
        )

    @pydantic.model_validator(mode="before")
    @classmethod
    def handle_deprecated_fields(cls, data: ArgsKwargs) -> ArgsKwargs:
        def _warn_and_rename_kwarg(data: ArgsKwargs, old_name: str, new_name: str) -> ArgsKwargs:
            assert data.kwargs is not None
            if old_name in data.kwargs:
                warnings.warn(
                    f"Field '{old_name}' is deprecated and will be removed in a future release. "
                    f"Please use '{new_name}' instead",
                    category=FutureWarning,
                    stacklevel=4,
                )
                data.kwargs[new_name] = data.kwargs.pop(old_name)
            return data

        if data.kwargs is not None:
            if "lead_time" in data.kwargs:
                # 'lead_time' was never used before, quietly ignore
                data.kwargs.pop("lead_time")
            for old_name, new_name in DEPRECATED_TASK_FIELDS.items():
                data = _warn_and_rename_kwarg(data, old_name, new_name)
        return data

    @property
    def freq(self) -> str:
        """Pandas string corresponding to the frequency of the time series in the dataset.

        See [pandas documentation](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#cutoff-aliases)
        for the list of possible values.

        This attribute is available after the dataset is loaded with `load_full_dataset`, `iter_windows` or `get_window`.
        """
        if self._freq is None:
            raise ValueError("Please load dataset first with `task.load_full_dataset()`")
        return self._freq

    @property
    def dynamic_columns(self) -> list[str]:
        """List of dynamic covariates available in the task. Does not include the target columns."""
        return self.known_dynamic_columns + self.past_dynamic_columns

    def _load_dataset(
        self,
        storage_options: dict | None = None,
        trust_remote_code: bool | None = None,
        num_proc: int = DEFAULT_NUM_PROC,
    ) -> datasets.Dataset:
        """Load the raw dataset and apply initial preprocessing based on the Task definition."""
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
        # Since we loaded with split=TRAIN and streaming=False, ds is a datasets.Dataset object
        assert isinstance(ds, datasets.Dataset)
        ds.set_format("numpy")

        required_columns = self.known_dynamic_columns + self.past_dynamic_columns + self.static_columns
        if self.generate_univariate_targets_from is None:
            required_columns += self.target_columns
        elif self.generate_univariate_targets_from == ALL_AVAILABLE_COLUMNS:
            pass
        else:
            required_columns += self.generate_univariate_targets_from

        utils.validate_time_series_dataset(
            ds,
            id_column=self.id_column,
            timestamp_column=self.timestamp_column,
            required_columns=required_columns,
            num_proc=num_proc,
        )

        # Create separate instances from columns listed in `generate_univariate_targets_from`
        if self.generate_univariate_targets_from is not None:
            if self.generate_univariate_targets_from == ALL_AVAILABLE_COLUMNS:
                generate_univariate_targets_from = [
                    col
                    for col, feat in ds.features.items()
                    if isinstance(feat, datasets.Sequence) and col != self.timestamp_column
                ]
            else:
                generate_univariate_targets_from = self.generate_univariate_targets_from
            assert isinstance(self.target, str)
            ds = utils.generate_univariate_targets_from_multivariate(
                ds,
                id_column=self.id_column,
                new_target_column=self.target,
                generate_univariate_targets_from=generate_univariate_targets_from,
                num_proc=num_proc,
            )

        # Ensure that IDs are sorted alphabetically for consistent ordering
        if ds.features[self.id_column].dtype != "string":
            ds = ds.cast_column(self.id_column, datasets.Value("string"))
        ds = ds.sort(self.id_column)
        self._freq = pd.infer_freq(ds[0][self.timestamp_column])
        if self._freq is None:
            raise ValueError("Dataset contains irregular timestamps")

        available_dynamic_columns, available_static_columns = utils.infer_column_types(
            ds, id_column=self.id_column, timestamp_column=self.timestamp_column
        )
        missing_dynamic = set(self.dynamic_columns) - set(available_dynamic_columns)
        if len(missing_dynamic) > 0:
            raise ValueError(f"Dynamic columns not found in dataset: {sorted(missing_dynamic)}")
        missing_static = set(self.static_columns) - set(available_static_columns)
        if len(missing_static) > 0:
            raise ValueError(f"Static columns not found in dataset: {sorted(missing_static)}")
        self._dataset_fingerprint = utils.generate_fingerprint(ds)
        return ds

    @property
    def dataset_info(self) -> dict:
        return {
            "id_column": self.id_column,
            "timestamp_column": self.timestamp_column,
            "target": self.target,
            "static_columns": self.static_columns,
            "dynamic_columns": self.dynamic_columns,
            "known_dynamic_columns": self.known_dynamic_columns,
            "past_dynamic_columns": self.past_dynamic_columns,
        }

    @property
    def predictions_schema(self) -> datasets.Features:
        """Describes the format that the predictions must follow.

        Forecast must always include the key `"predictions"` corresponding to the point forecast.

        The predictions must also include a key for each of the `quantile_levels`.
        For example, if `quantile_levels = [0.1, 0.9]`, then keys `"0.1"` and `"0.9"` must be included in the forecast.
        """
        predictions_length = self.horizon
        predictions_schema = {
            PREDICTIONS: datasets.Sequence(datasets.Value("float64"), length=predictions_length),
        }
        for q in sorted(self.quantile_levels):
            predictions_schema[str(q)] = datasets.Sequence(datasets.Value("float64"), length=predictions_length)
        return datasets.Features(predictions_schema)

    def clean_and_validate_predictions(
        self, predictions: datasets.DatasetDict | dict[str, list[dict]] | datasets.Dataset | list[dict]
    ) -> datasets.DatasetDict:
        """Convert predictions for a single window into the format needed for computing the metrics.

        The following formats are supported for both multivariate and univariate tasks:

        - `DatasetDict`: Must contain a single key for each target in `task.target_columns`. Each value in
            the dict must be a `datasets.Dataset` with schema compatible with `task.predictions_schema`. This is the
            recommended format for providing predictions.
        - `dict[str, list[dict]]`: A dictionary with one key for each target in `task.target_columns`. Each value in
            the dict must be a list of dictionaries, each dict following the schema in `task.predictions_schema`.

        Additionally for univariate tasks, the following formats are supported:

        - `datasets.Dataset`: A single `datasets.Dataset` with schema compatible with `task.predictions_schema`.
        - `list[dict]`: A list of dictionaries, where each dict follows the schema in `task.predictions_schema`.

        Returns
        -------
        predictions :
            A `DatasetDict` where each key is the name of the target column and the corresponding value is a
            `datasets.Dataset` with the predictions.
        """

        def _to_dataset(preds: datasets.Dataset | list[dict]) -> datasets.Dataset:
            if isinstance(preds, list):
                try:
                    preds = datasets.Dataset.from_list(list(preds))
                except Exception:
                    raise ValueError(
                        "`datasets.Dataset.from_list(predictions)` failed. Please convert predictions to `datasets.Dataset` format."
                    )
            if not isinstance(preds, datasets.Dataset):
                raise ValueError(f"predictions must be of type `datasets.Dataset` (received {type(preds)})")
            return preds

        if isinstance(predictions, datasets.DatasetDict):
            pass
        elif isinstance(predictions, dict):
            predictions = datasets.DatasetDict({col: _to_dataset(preds) for col, preds in predictions.items()})
        elif isinstance(predictions, (list, datasets.Dataset)):
            predictions = datasets.DatasetDict({self.target_columns[0]: _to_dataset(predictions)})
        else:
            raise ValueError(
                f"Expected predictions to be a `DatasetDict`, `Dataset`, `list` or `dict` (got {type(predictions)})"
            )
        if missing_columns := set(self.target_columns) - set(predictions.keys()):
            raise ValueError(f"Missing predictions for columns {missing_columns} (got {sorted(predictions.keys())})")
        predictions = predictions.cast(self.predictions_schema).with_format("numpy")
        for target_column, predictions_for_column in predictions.items():
            self._assert_all_columns_finite(predictions_for_column)
        return predictions

    @staticmethod
    def _assert_all_columns_finite(predictions: datasets.Dataset) -> None:
        for col in predictions.column_names:
            nan_row_idx, _ = np.where(~np.isfinite(np.array(predictions[col])))
            if len(nan_row_idx) > 0:
                raise ValueError(
                    "Predictions contain NaN or Inf values. "
                    f"First invalid value encountered in column {col} for item {nan_row_idx[0]}:\n"
                    f"{predictions[int(nan_row_idx[0])]}"
                )

    def evaluation_summary(
        self,
        predictions_per_window: Iterable[datasets.Dataset | list[dict] | datasets.DatasetDict | dict[str, list[dict]]],
        model_name: str,
        training_time_s: float | None = None,
        inference_time_s: float | None = None,
        trained_on_this_dataset: bool = False,
        extra_info: dict | None = None,
    ) -> dict[str, Any]:
        """Get a summary of the model performance for the given forecasting task.

        Parameters
        ----------
        predictions_per_window : Iterable[datasets.Dataset | list[dict] | datasets.DatasetDict | dict[str, list[dict]]]
            Predictions generated by the model for each evaluation window in the task.

            The length of `predictions_per_window` must be equal to `task.num_windows`.

            The predictions for each window must be formatted as described in [clean_and_validate_predictions][fev.Task.clean_and_validate_predictions].
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
            - `test_error` - value of the `task.eval_metric` averaged over all evaluation windows (lower is better)
            - all `Task` attributes obtained via `task.to_dict()`.
            - values of `task.extra_metrics` averaged all evaluation windows
            - `dataset_fingerprint` - fingerprint of the dataset generated by the HF `datasets` library
            - `trained_on_this_dataset` - whether the model was trained on the dataset used in the task
            - `fev_version` - version of the `fev` package used to obtain the summary
        """
        summary: dict[str, Any] = {"model_name": model_name}
        summary.update(self.to_dict())
        metrics = [get_metric(m) for m in [self.eval_metric] + self.extra_metrics]
        eval_metric = metrics[0]

        metrics_per_window = {metric.name: [] for metric in metrics}
        if isinstance(predictions_per_window, (datasets.Dataset, datasets.DatasetDict, dict)):
            raise ValueError(
                f"predictions_per_window must be iterable (e.g., a list) but got {type(predictions_per_window)}"
            )
        # Use strict=True to raise error if num_predictions does not match num_windows
        for predictions, window in zip(predictions_per_window, self.iter_windows(), strict=True):
            metric_scores = window.compute_metrics(
                self.clean_and_validate_predictions(predictions),
                metrics=metrics,
                seasonality=self.seasonality,
                quantile_levels=self.quantile_levels,
            )
            for metric, value in metric_scores.items():
                metrics_per_window[metric].append(value)
        metrics_averaged = {metric_name: float(np.mean(values)) for metric_name, values in metrics_per_window.items()}
        summary.update(
            {
                "test_error": metrics_averaged[eval_metric.name],
                "training_time_s": training_time_s,
                "inference_time_s": inference_time_s,
                "dataset_fingerprint": self._dataset_fingerprint,
                "trained_on_this_dataset": trained_on_this_dataset,
                "fev_version": FEV_VERSION,
                **metrics_averaged,
            }
        )
        if extra_info is not None:
            summary.update(extra_info)
        return summary

    @property
    def is_multivariate(self) -> bool:
        """Returns `True` if `task.target` is a `list`, `False` otherwise."""
        return isinstance(self.target, list)

    @property
    def target_columns(self) -> list[str]:
        """A list including names of all target columns for this task.

        Unlike `task.target` that can be a string or a list, `task.target_columns` is always a list of strings.
        """
        if isinstance(self.target, list):
            return self.target
        else:
            return [self.target]


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


def _has_enough_past_and_future_observations(
    record: dict,
    timestamp_column: str,
    horizon: int,
    cutoff: str | int,
    min_context_length: int,
) -> bool:
    """Return True if time series has >= `min_context_length` observations before `cutoff` and >= `horizon` observations after."""
    timestamps = record[timestamp_column]
    if isinstance(cutoff, str):
        before = timestamps[timestamps <= np.datetime64(cutoff)]
        after = timestamps[timestamps > np.datetime64(cutoff)]
    else:
        before = timestamps[:cutoff]
        after = timestamps[cutoff:]
    return len(before) >= min_context_length and len(after) >= horizon

import reprlib
import warnings
from collections import defaultdict

import datasets
import multiprocess as mp
import pandas as pd

from .constants import DEFAULT_NUM_PROC

__all__ = [
    "convert_long_df_to_hf_dataset",
    "infer_column_types",
    "validate_time_series_dataset",
    "generate_univariate_targets_from_multivariate",
]


def validate_time_series_dataset(
    dataset: datasets.Dataset,
    id_column: str = "id",
    timestamp_column: str = "timestamp",
    num_proc: int = DEFAULT_NUM_PROC,
    required_columns: list[str] | None = None,
    num_records_to_validate: int | None = None,
) -> None:
    """Ensure that `datasets.Dataset` object is a valid time series dataset.

    This methods validates the following assumptions:

    - `id_column` is present and has type `Value('string')`
    - all values in the `id_column` are unique
    - `timestamp_column` is present and has type `Sequence(Value('timestamp'))`
    - at least 1 dynamic column is present

    Following checks are performed for the first `num_records_to_validate` records in the dataset (or for all records
    if `num_records_to_validate=None`)

    - timestamps have a regular frequency that can be inferred with pandas.infer_freq
    - values in all dynamic columns have same length as timestamp_column

    Parameters
    ----------
    dataset
        Dataset that must be validated.
    id_column
        Name of the column containing the unique ID of each time series.
    timestamp_column
        Name of the column containing the timestamp of time series observations.
    """
    if required_columns is None:
        required_columns = []
    required_columns += [id_column, timestamp_column]
    missing_columns = set(required_columns).difference(set(dataset.column_names))
    if len(missing_columns) > 0:
        raise AssertionError(
            f"Following {len(missing_columns)} columns are missing from the dataset: {reprlib.repr(missing_columns)}. "
            f"Available columns: {dataset.column_names}"
        )

    id_feature = dataset.features[id_column]
    if not isinstance(id_feature, datasets.Value):
        raise AssertionError(f"id_column {id_column} must have type Value")
    timestamp_feature = dataset.features[timestamp_column]
    if not (
        isinstance(timestamp_feature, datasets.Sequence) and timestamp_feature.feature.dtype.startswith("timestamp")
    ):
        raise AssertionError(f"timestamp_column {timestamp_column} must have type Sequence(Value('timestamp'))")

    if len(set(dataset[id_column])) != len(dataset[id_column]):
        raise AssertionError(f"ID column {id_column} must contain unique values for each record")

    dynamic_columns, static_columns = infer_column_types(
        dataset,
        id_column,
        timestamp_column,
    )

    if len(dynamic_columns) == 0:
        raise AssertionError("Dataset must contain at least a single dynamic column of type Sequence")

    if num_records_to_validate is not None:
        dataset = dataset.select(range(num_records_to_validate))
    dataset.map(
        _validate_dynamic_columns,
        num_proc=min(num_proc, len(dataset)),
        desc="Validating dataset format",
        fn_kwargs={"id_column": id_column, "timestamp_column": timestamp_column, "dynamic_columns": dynamic_columns},
    )


def _validate_dynamic_columns(record: dict, id_column: str, timestamp_column: str, dynamic_columns: list[str]) -> None:
    """Validate dynamic columns for a single record."""
    timestamps = record[timestamp_column]
    if pd.infer_freq(timestamps) is None:
        raise AssertionError(f"pd.infer_freq failed to infer timestamp frequency for record {record[id_column]}.")
    for col in dynamic_columns:
        if len(record[col]) != len(timestamps):
            raise AssertionError(
                f"Length of dynamic column {col} doesn't match the length of the timestamp column for record {record[id_column]}"
            )


def infer_column_types(
    dataset: datasets.Dataset,
    id_column: str,
    timestamp_column: str,
) -> tuple[list[str], list[str]]:
    """Infer the types of columns in a time series dataset.

    Columns that have type `datasets.Sequence` are interpreted as dynamic features, and all remaining columns except
    `id_column` and `timestamp_column` are interpreted as static features.

    Parameters
    ----------
    dataset
        Time series dataset.
    id_column : str
        Name of the column with the unique identifier of each time series.
    timestamp_column : str
        Name of the column with the timestamps of the observations.

    Returns
    -------
    dynamic_columns : List[str]
        Names of columns that contain dynamic (time-varying) features.
    static_columns : List[str]
        Names of columns that contain static (time-independent) features.
    """
    dynamic_columns = []
    static_columns = []
    for col_name, col_type in dataset.features.items():
        if col_name not in [id_column, timestamp_column]:
            if isinstance(col_type, datasets.Sequence):
                dynamic_columns.append(col_name)
            else:
                static_columns.append(col_name)
    return dynamic_columns, static_columns


class PatchedDownloadConfig(datasets.DownloadConfig):
    # Fixes a bug that prevents `load_dataset` from loading datasets from S3.
    # See https://github.com/huggingface/datasets/issues/6598
    def __post_init__(self, use_auth_token):
        if use_auth_token != "deprecated":
            self.token = use_auth_token


def convert_long_df_to_hf_dataset(
    df: pd.DataFrame,
    id_column: str = "id",
    timestamp_column: str = "timestamp",
    static_columns: list[str] | None = None,
    num_proc: int = DEFAULT_NUM_PROC,
) -> datasets.Dataset:
    """Convert a long-format pandas DataFrame to a Hugging Face datasets.Dataset object.

    Parameters
    ----------
    df:
        Long-format DataFrame containing the data.
    id_column
        Name of the column containing the unique ID of each time series.
    timestamp_column
        Name of the column containing the timestamp of time series observations.
    static_columns
        Names of columns that contain static (time-independent) features.
    num_proc
        Number of processes used to parallelize the computation.
    """
    df[id_column] = df[id_column].astype(str)
    df[timestamp_column] = pd.to_datetime(df[timestamp_column])
    df = df.sort_values(by=[id_column, timestamp_column])

    if static_columns is None:
        static_columns = []
    static_columns = [id_column] + static_columns

    def process_entry(group: pd.DataFrame) -> dict:
        static = group[static_columns].iloc[0].to_dict()
        dynamic = group.drop(columns=static_columns).to_dict("list")
        return {**static, **dynamic}

    with mp.Pool(processes=num_proc) as pool:
        entries = pool.map(process_entry, [group for _, group in df.groupby(id_column, sort=False)])
    return datasets.Dataset.from_list(entries)


def generate_fingerprint(dataset: datasets.Dataset, num_rows_to_check: int = 3) -> str | None:
    """Generate a fingerprint for the PyArrow Table backing the Dataset.

    Unlike `datasets.fingerprint.generate_fingerprint`, this method only considers the underlying PyArrow Table, and
    not other dataset attributes such as DatasetInfo or the last modified timestamp.

    The fingerprint depends on the first and last `num_rows_to_check` of the dataset, and the metadata such as
    `schema`, `nbytes` and `num_rows`.

    Parameters
    ----------
    dataset : datasets.Dataset
        Dataset for which to generate the fingerprint.
    num_rows_to_check : int, default 3
        Number of rows at the start and the end of the dataset to check when generating the fingerprint.
    """
    if not isinstance(dataset, datasets.Dataset):
        raise ValueError(f"Expected a datasets.Dataset object (got type {type(dataset)})")
    try:
        hasher = datasets.fingerprint.Hasher()
        # Compute hash of the first and last `num_rows` rows of the data
        hasher.update(dataset.with_format("arrow")[:num_rows_to_check])
        hasher.update(dataset.with_format("arrow")[-num_rows_to_check:])
        table = dataset._data
        # Update hash based on the dataset schema and size
        for attr in [table.schema, table.nbytes, table.num_rows]:
            hasher.update(attr)
        return hasher.hexdigest()
    except Exception as e:
        # In case the private API `Dataset._data` breaks at some point
        warnings.warn(f"generate_fingerprint failed with exception '{str(e)}'")
        return None


def _expand_target_columns(
    batch: dict, id_column: str, target_column: str, generate_univariate_targets_from: list[str]
) -> dict:
    """Create a separate record for each column listed in generate_univariate_targets_from.

    It is required to set batched=True when using method in `dataset.map`.
    """
    expanded_batch = defaultdict(list)
    batch_size = len(batch[id_column])
    for i in range(batch_size):
        for target_col in generate_univariate_targets_from:
            for key in batch.keys():
                if key not in generate_univariate_targets_from:
                    value = batch[key][i]
                    if key == id_column:
                        value = value + "_" + target_col
                    expanded_batch[key].append(value)
            expanded_batch[target_column].append(batch[target_col][i])
    return dict(expanded_batch)


def generate_univariate_targets_from_multivariate(
    dataset: datasets.Dataset,
    id_column: str,
    new_target_column: str,
    generate_univariate_targets_from: list[str],
    num_proc: int = DEFAULT_NUM_PROC,
):
    """Convert each multivariate time series in the dataset into multiple univariate series.

    Creates separate univariate time series from specified columns by expanding each
    record into multiple records with modified IDs (format: "{id}_{column_name}").

    Parameters
    ----------
    ds
        Input multivariate time series dataset.
    id_column
        Column containing unique time series identifiers.
    new_target_column
        Output column name for target values.
    generate_univariate_targets_from
        Columns to convert into separate univariate series.
    num_proc
        Number of processes for parallel processing.
    """
    return dataset.map(
        _expand_target_columns,
        batched=True,
        fn_kwargs=dict(
            id_column=id_column,
            target_column=new_target_column,
            generate_univariate_targets_from=generate_univariate_targets_from,
        ),
        remove_columns=generate_univariate_targets_from,
        num_proc=num_proc,
    )


def combine_univariate_predictions_to_multivariate(
    predictions: datasets.Dataset | list[dict] | datasets.DatasetDict | dict[str, list[dict]],
    target_columns_list: list[str],
) -> datasets.DatasetDict:
    """Combine univariate predictions back into multivariate format.

    Assumes predictions are ordered by cycling through target columns. For example: if target_columns_list = ["X", "Y"],
    predictions should be ordered as [item1_X, item1_Y, item2_X, item2_Y, ...].

    Returns a DatasetDict with one key per target column.
    """
    if isinstance(predictions, (dict, datasets.DatasetDict)):
        assert len(predictions) == 1, "Univariate predictions must contain a single key/value"
        predictions = next(iter(predictions.values()))
    if isinstance(predictions, list):
        try:
            predictions = datasets.Dataset.from_list(predictions)
        except Exception:
            raise ValueError(
                "`datasets.Dataset.from_list(predictions)` failed. Please convert predictions to `datasets.Dataset` format."
            )
    assert isinstance(predictions, datasets.Dataset), "predictions must be a datasets.Dataset object"
    assert len(predictions) % len(target_columns_list) == 0, (
        "Number of predictions must be divisible by the number of target columns"
    )
    prediction_dict = {}
    for i, col in enumerate(target_columns_list):
        prediction_dict[col] = predictions.select(range(i, len(predictions), len(target_columns_list)))
    return datasets.DatasetDict(prediction_dict)

import reprlib
import warnings
from collections import defaultdict

import datasets
import multiprocess as mp
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from .constants import DEFAULT_NUM_PROC

__all__ = [
    "convert_long_df_to_hf_dataset",
    "infer_column_types",
    "validate_time_series_dataset",
    "generate_univariate_targets_from_multivariate",
    "combine_univariate_predictions_to_multivariate",
    "past_future_split",
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
    dataset = dataset.with_format("numpy")
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

    dynamic_columns, static_columns = infer_column_types(dataset, id_column, timestamp_column)

    if len(dynamic_columns) == 0:
        raise AssertionError("Dataset must contain at least a single dynamic column of type Sequence")

    if num_records_to_validate is not None:
        dataset = dataset.select(range(num_records_to_validate))

    dataset.map(
        _validate_frequency,
        num_proc=min(num_proc, len(dataset)),
        desc="Validating dataset format",
        batched=True,
        input_columns=[timestamp_column],
    )

    # Ensure that for each row all entries in columns of type Sequence have the same length
    table = dataset.data.table
    expected_lengths = pc.list_value_length(table[timestamp_column])
    for col in dynamic_columns:
        if not pc.list_value_length(table[col]) == expected_lengths:
            raise AssertionError(
                f"Lengths of entries in {col} does not match the lengths in {timestamp_column} for all records"
            )


def _validate_frequency(batch_of_timestamps: list) -> None:
    """Assert that the frequency can be inferred for each record."""
    for timestamps in batch_of_timestamps:
        if pd.infer_freq(timestamps) is None:
            raise AssertionError("pd.infer_freq failed to infer timestamp frequency.")


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
        num_proc=min(num_proc, len(dataset)),
    )


def combine_univariate_predictions_to_multivariate(
    predictions: datasets.Dataset | list[dict] | datasets.DatasetDict | dict[str, list[dict]],
    target_columns: list[str],
) -> datasets.DatasetDict:
    """Combine univariate predictions back into multivariate format.

    Assumes predictions are ordered by cycling through target columns. For example: if `target_columns = ["X", "Y"]`,
    predictions should be ordered as `[item1_X, item1_Y, item2_X, item2_Y, ...]`.

    Parameters
    ----------
    predictions
        Univariate predictions for a single evaluation window.

        For the list of accepted types, see [`Task.clean_and_validate_predictions`][fev.Task.clean_and_validate_predictions].
    target_columns
        List of target columns in the original `Task` / `EvaluationWindow`.

    Returns
    -------
    datasets.DatasetDict
        Predictions for the evaluation window converted to multivariate format.
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
    assert len(predictions) % len(target_columns) == 0, (
        "Number of predictions must be divisible by the number of target columns"
    )
    prediction_dict = {}
    for i, col in enumerate(target_columns):
        prediction_dict[col] = predictions.select(range(i, len(predictions), len(target_columns)))
    return datasets.DatasetDict(prediction_dict)


def _compute_cutoff_indices(
    timestamps_col: pa.Array, cutoff: int | str, lengths: np.ndarray, offsets: np.ndarray
) -> np.ndarray:
    """Compute cutoff index for each time series (number of elements at or before cutoff).

    For string cutoffs (timestamps), counts elements where timestamp <= cutoff.
    For integer cutoffs, treats as array index (negative indices count from end).
    """
    if isinstance(cutoff, str):
        timestamps_flat = pc.list_flatten(timestamps_col)
        cutoff_scalar = pc.cast(pa.scalar(cutoff), timestamps_flat.type)
        mask = pc.less_equal(timestamps_flat, cutoff_scalar)
        cumsum = np.concatenate([[0], np.cumsum(mask.to_numpy(zero_copy_only=False))])
        return cumsum[offsets[1:]] - cumsum[offsets[:-1]]
    else:
        cutoff_indices = np.where(cutoff >= 0, cutoff, lengths + cutoff)
        return np.clip(cutoff_indices, 0, lengths)


def _filter_by_length(
    dataset: datasets.Dataset,
    cutoff_indices: np.ndarray,
    lengths: np.ndarray,
    min_context_length: int,
    horizon: int,
) -> tuple[datasets.Dataset, np.ndarray, np.ndarray]:
    """Filter dataset to keep only series with sufficient context and future observations.

    Returns filtered (dataset, cutoff_indices, lengths) tuple.
    """
    valid_mask = (cutoff_indices >= min_context_length) & ((lengths - cutoff_indices) >= horizon)

    if valid_mask.all():
        return dataset, cutoff_indices, lengths

    valid_indices = np.nonzero(valid_mask)[0]
    filtered_dataset = dataset.select(valid_indices).flatten_indices()
    return filtered_dataset, cutoff_indices[valid_mask], lengths[valid_mask]


def _build_sliced_dataset(
    dataset: datasets.Dataset,
    table: pa.Table,
    sequence_columns: list[str],
    offsets: np.ndarray,
    slice_start: np.ndarray,
    slice_end: np.ndarray,
) -> datasets.Dataset:
    """Build a new dataset with sequence columns sliced to [slice_start, slice_end) per row.

    Uses an event-based approach to efficiently build a boolean mask for the flattened
    sequences, then reconstructs list arrays with new offsets.
    """
    slice_lengths = slice_end - slice_start
    total_elements = offsets[-1]

    # Build boolean mask using +1/-1 events at slice boundaries
    events = np.zeros(total_elements + 1, dtype=np.int8)
    global_starts = offsets[:-1] + slice_start
    global_ends = offsets[:-1] + slice_end
    np.add.at(events, global_starts, 1)
    np.add.at(events, global_ends, -1)
    mask = pa.array(np.cumsum(events[:total_elements], dtype=np.int8).view(bool))

    # Compute new offsets for the sliced arrays
    new_offsets = pa.array(np.concatenate([[0], np.cumsum(slice_lengths)]))

    # Build new columns
    new_columns = {}
    for col_name in dataset.column_names:
        col = table[col_name].combine_chunks()
        if col_name in sequence_columns:
            filtered_values = pc.filter(pc.list_flatten(col), mask)
            if isinstance(filtered_values, pa.ChunkedArray):
                filtered_values = filtered_values.combine_chunks()
            new_columns[col_name] = pa.ListArray.from_arrays(new_offsets, filtered_values)
        else:
            new_columns[col_name] = col

    new_table = pa.table(new_columns)
    new_dataset = datasets.Dataset(new_table, fingerprint=datasets.fingerprint.generate_random_fingerprint())
    return new_dataset.with_format(dataset.format["type"])


def past_future_split(
    dataset: datasets.Dataset,
    timestamp_column: str,
    cutoff: int | str,
    horizon: int,
    min_context_length: int,
    max_context_length: int | None = None,
    return_past: bool = True,
    return_future: bool = True,
) -> tuple[datasets.Dataset | None, datasets.Dataset | None]:
    """Filter and slice time series sequences in a single efficient pass.

    Computes cutoff indices once and returns both past and future data, filtering
    out series that don't have sufficient data. This is more efficient than calling
    separate filtering and slicing methods.

    Parameters
    ----------
    dataset
        Time series dataset to process.
    timestamp_column
        Name of the column containing timestamps.
    cutoff
        Cutoff point as integer index or timestamp string. Negative indices count from end.
    horizon
        Number of observations to include in future data.
    min_context_length
        Minimum required observations before cutoff for filtering.
    max_context_length
        Maximum observations to include in past data. If None, includes all data before cutoff.
    return_past
        Whether to compute and return past data. If False, past_data will be None.
    return_future
        Whether to compute and return future data. If False, future_data will be None.

    Returns
    -------
    tuple[datasets.Dataset | None, datasets.Dataset | None]
        Tuple of (past_data, future_data) with sequence columns sliced appropriately.
        Either may be None if the corresponding return flag is False.
    """
    # Flatten indices if dataset has been sorted/filtered, so row order in dataset
    # matches the physical order in the underlying Arrow table
    if getattr(dataset, "_indices", None) is not None:
        dataset = dataset.flatten_indices()

    table = dataset.data.table
    timestamps_col = table[timestamp_column].combine_chunks()
    sequence_columns = [col for col, feat in dataset.features.items() if isinstance(feat, datasets.Sequence)]
    lengths = pc.list_value_length(timestamps_col).to_numpy()
    offsets = np.concatenate([[0], np.cumsum(lengths)])

    # Compute cutoff indices
    cutoff_indices = _compute_cutoff_indices(timestamps_col, cutoff, lengths, offsets)

    # Filter series with insufficient context or future observations
    dataset, cutoff_indices, lengths = _filter_by_length(dataset, cutoff_indices, lengths, min_context_length, horizon)

    # Re-extract table and recompute offsets after filtering
    table = dataset.data.table
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    n = len(dataset)

    # Past data: [max(0, cutoff - max_context_length), cutoff)
    # After filtering: cutoff_indices >= min_context_length, so cutoff_indices is valid
    past_data = None
    if return_past:
        if max_context_length is not None:
            past_start = np.maximum(cutoff_indices - max_context_length, 0)
        else:
            past_start = np.zeros(n, dtype=np.int64)
        past_data = _build_sliced_dataset(dataset, table, sequence_columns, offsets, past_start, cutoff_indices)

    # Future data: [cutoff, cutoff + horizon)
    # After filtering: lengths - cutoff_indices >= horizon, so cutoff + horizon <= lengths
    future_data = None
    if return_future:
        future_end = cutoff_indices + horizon
        future_data = _build_sliced_dataset(dataset, table, sequence_columns, offsets, cutoff_indices, future_end)

    return past_data, future_data

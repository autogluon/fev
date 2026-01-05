import datasets
import numpy as np
import pyarrow as pa


def filter_short_series(
    dataset: datasets.Dataset,
    timestamp_column: str,
    cutoff: int | str,
    min_context_length: int,
    horizon: int,
) -> datasets.Dataset:
    """Vectorized filtering of time series that are too short."""
    table = dataset.data.table
    timestamps_col = table[timestamp_column].combine_chunks()
    offsets = timestamps_col.offsets.to_numpy()
    lengths = np.diff(offsets)

    if isinstance(cutoff, str):
        cumsum_mask = np.concatenate([[0], np.cumsum(timestamps_col.values.to_numpy() <= np.datetime64(cutoff))])
        before_count = cumsum_mask[offsets[1:]] - cumsum_mask[offsets[:-1]]
        after_count = lengths - before_count
    else:
        cutoff_indices = np.where(cutoff >= 0, cutoff, lengths + cutoff)
        before_count = np.clip(cutoff_indices, 0, lengths)
        after_count = lengths - before_count

    valid = (before_count >= min_context_length) & (after_count >= horizon)
    return dataset.select(np.where(valid)[0])


def slice_sequence_columns(
    dataset: datasets.Dataset,
    timestamp_column: str,
    cutoff: int | str,
    max_context_length: int | None = None,
    horizon: int | None = None,
) -> datasets.Dataset:
    """Fast slicing of Sequence columns using vectorized PyArrow operations.

    Selects data before cutoff (if max_context_length is provided) or after cutoff (if horizon is provided).
    """
    table = dataset.data.table
    columns_to_slice = [col for col, feat in dataset.features.items() if isinstance(feat, datasets.Sequence)]

    # Compute cutoff indices
    if isinstance(cutoff, str):
        timestamps_col = table[timestamp_column].combine_chunks()
        offsets = timestamps_col.offsets.to_numpy()
        cumsum_mask = np.concatenate([[0], np.cumsum(timestamps_col.values.to_numpy() <= np.datetime64(cutoff))])
        cutoff_indices = cumsum_mask[offsets[1:]] - cumsum_mask[offsets[:-1]]
    else:
        offsets = table[columns_to_slice[0]].combine_chunks().offsets.to_numpy()
        cutoff_indices = np.full(len(table), cutoff, dtype=np.int64)

    # Determine slice bounds based on whether we want past or future data
    if horizon is not None:
        # Future data: from cutoff to cutoff + horizon
        start_indices = cutoff_indices
        end_indices = cutoff_indices + horizon
    else:
        # Past data: up to cutoff, optionally limited by max_context_length
        start_indices = cutoff_indices - max_context_length if max_context_length else None
        end_indices = cutoff_indices

    # Compute slice bounds
    lengths = np.diff(offsets)
    slice_start = (
        np.zeros(len(table), dtype=np.int64)
        if start_indices is None
        else np.clip(np.where(start_indices >= 0, start_indices, lengths + start_indices), 0, lengths)
    )
    slice_end = np.clip(np.where(end_indices >= 0, end_indices, lengths + end_indices), 0, lengths)
    valid = slice_start < slice_end

    # Compute mask using cumsum trick
    events = np.zeros(offsets[-1] + 1, dtype=np.int8)
    events[offsets[:-1][valid] + slice_start[valid]] = 1
    events[offsets[:-1][valid] + slice_end[valid]] = -1
    mask = np.cumsum(events)[:-1].astype(bool)
    new_offsets = np.concatenate([[0], np.cumsum(np.where(valid, slice_end - slice_start, 0))])

    # Apply mask to all columns
    new_columns = {}
    for col_name in table.column_names:
        if col_name in columns_to_slice:
            list_array = table[col_name].combine_chunks()
            new_columns[col_name] = pa.ListArray.from_arrays(
                pa.array(new_offsets, type=pa.int32()),
                pa.array(list_array.values.to_numpy(zero_copy_only=False)[mask], type=list_array.values.type),
            )
        else:
            new_columns[col_name] = table[col_name]

    return datasets.Dataset(pa.table(new_columns), fingerprint=datasets.fingerprint.generate_random_fingerprint())

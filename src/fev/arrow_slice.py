import datasets
import numpy as np
import pyarrow as pa


def slice_sequence_columns(
    dataset: datasets.Dataset,
    timestamp_column: str,
    cutoff: int | str,
    start_offset: int | None = None,
    length: int | None = None,
) -> datasets.Dataset:
    """Fast slicing of Sequence columns using vectorized PyArrow operations."""
    table = dataset.data.table
    columns_to_slice = [col for col, feat in dataset.features.items() if isinstance(feat, datasets.Sequence)]

    # Compute end indices based on cutoff
    if isinstance(cutoff, str):
        timestamps_col = table[timestamp_column].combine_chunks()
        offsets = timestamps_col.offsets.to_numpy()
        timestamps_flat = timestamps_col.values.to_numpy()
        cutoff_ts = np.datetime64(cutoff)
        cumsum_mask = np.concatenate([[0], np.cumsum(timestamps_flat <= cutoff_ts)])
        end_indices = cumsum_mask[offsets[1:]] - cumsum_mask[offsets[:-1]]
    else:
        offsets = table[columns_to_slice[0]].combine_chunks().offsets.to_numpy()
        end_indices = np.full(len(table), cutoff, dtype=np.int64)

    if start_offset is not None:
        start_indices = end_indices + start_offset
    else:
        start_indices = None

    if length is not None:
        end_indices = (start_indices if start_indices is not None else np.zeros(len(table), dtype=np.int64)) + length

    # Compute slice bounds once for all columns
    lengths = np.diff(offsets)
    slice_start = (
        np.zeros(len(table), dtype=np.int64)
        if start_indices is None
        else np.clip(np.where(start_indices >= 0, start_indices, lengths + start_indices), 0, lengths)
    )
    slice_end = np.clip(np.where(end_indices >= 0, end_indices, lengths + end_indices), 0, lengths)
    valid = slice_start < slice_end

    # Compute mask once using cumsum trick
    events = np.zeros(offsets[-1] + 1, dtype=np.int8)
    events[offsets[:-1][valid] + slice_start[valid]] += 1
    events[offsets[:-1][valid] + slice_end[valid]] -= 1
    mask = np.cumsum(events)[:-1].astype(bool)
    new_lengths = np.where(valid, slice_end - slice_start, 0)
    new_offsets = np.concatenate([[0], np.cumsum(new_lengths)])

    # Apply mask to all columns
    new_columns = {}
    for col_name in table.column_names:
        if col_name not in columns_to_slice:
            new_columns[col_name] = table[col_name]
        else:
            list_array = table[col_name].combine_chunks()
            values = list_array.values.to_numpy()[mask]
            new_columns[col_name] = pa.ListArray.from_arrays(
                pa.array(new_offsets, type=pa.int32()), pa.array(values, type=list_array.values.type)
            )

    return datasets.Dataset(pa.table(new_columns), fingerprint=datasets.fingerprint.generate_random_fingerprint())

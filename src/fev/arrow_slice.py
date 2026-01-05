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
        cutoff_ts = np.datetime64(cutoff)
        timestamps_col = table[timestamp_column]
        end_indices = np.array(
            [np.searchsorted(timestamps_col[i].as_py(), cutoff_ts, side="right") for i in range(len(table))],
            dtype=np.int64,
        )
    else:
        end_indices = np.full(len(table), cutoff, dtype=np.int64)

    # Compute start indices
    if start_offset is not None:
        start_indices = end_indices + start_offset
    else:
        start_indices = None

    # Adjust end indices if length is specified
    if length is not None:
        end_indices = (start_indices if start_indices is not None else np.zeros(len(table), dtype=np.int64)) + length

    # Slice all list columns
    new_columns = {}
    for col_name in table.column_names:
        if col_name not in columns_to_slice:
            new_columns[col_name] = table[col_name]
            continue

        list_array = table[col_name].combine_chunks()
        offsets = list_array.offsets.to_numpy()
        lengths = np.diff(offsets)

        slice_start = (
            np.zeros(len(table), dtype=np.int64)
            if start_indices is None
            else np.clip(np.where(start_indices >= 0, start_indices, lengths + start_indices), 0, lengths)
        )
        slice_end = np.clip(np.where(end_indices >= 0, end_indices, lengths + end_indices), 0, lengths)

        valid = slice_start < slice_end
        if not np.any(valid):
            new_columns[col_name] = pa.array([[] for _ in range(len(table))], type=list_array.type)
            continue

        # Cumsum trick for fast slicing
        events = np.zeros(offsets[-1] + 1, dtype=np.int8)
        events[offsets[:-1][valid] + slice_start[valid]] += 1
        events[offsets[:-1][valid] + slice_end[valid]] -= 1
        mask = np.cumsum(events)[:-1].astype(bool)

        values = list_array.values.to_numpy()[mask]
        new_lengths = np.where(valid, slice_end - slice_start, 0)
        new_offsets = np.concatenate([[0], np.cumsum(new_lengths)])

        new_columns[col_name] = pa.ListArray.from_arrays(
            pa.array(new_offsets, type=pa.int32()), pa.array(values, type=list_array.values.type)
        )

    return datasets.Dataset(datasets.arrow_dataset.InMemoryTable(pa.table(new_columns)))

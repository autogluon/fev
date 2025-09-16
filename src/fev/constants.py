from multiprocessing import cpu_count

import datasets

PREDICTIONS = "predictions"
DEFAULT_NUM_PROC = cpu_count()

TRAIN = datasets.Split.TRAIN
FUTURE = datasets.splits.NamedSplit("future")
TEST = datasets.Split.TEST

DEPRECATED_TASK_FIELDS = {
    "num_rolling_windows": "num_windows",
    "rolling_step_size": "window_step_size",
    "cutoff": "initial_cutoff",
    "target_column": "target",
    "multiple_target_columns": "generate_univariate_targets_from",
}

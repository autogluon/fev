from multiprocessing import cpu_count

PREDICTIONS = "predictions"
DEFAULT_NUM_PROC = cpu_count()

DEPRECATED_TASK_FIELDS = {
    "num_rolling_windows": "num_windows",
    "rolling_step_size": "window_step_size",
    "cutoff": "initial_cutoff",
    "target_column": "target",
    "multiple_target_columns": "generate_univariate_targets_from",
}

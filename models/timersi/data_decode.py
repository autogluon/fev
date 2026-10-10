"""Decode each needed Arrow column once per batch, preserving stock native inputs."""


def materialize_covariates(task, past, future, target_columns=None):
    targets = task.target_columns if target_columns is None else target_columns
    if isinstance(targets, str):
        targets = [targets]
    names = list(dict.fromkeys([*targets, *task.known_dynamic_columns, *task.past_dynamic_columns]))
    if hasattr(past, 'select_columns'):
        past = past.select_columns(names).to_dict()
    if future is not None and hasattr(future, 'select_columns'):
        future = future.select_columns(task.known_dynamic_columns).to_dict() if task.known_dynamic_columns else None
    return past, future


class CovariateDecodeSwitch:
    def __init__(self):
        self.enabled = False

    def install(self):
        import native
        original = native.prepare_covariates_for_timesfm3

        def prepare(task, imputation, past_batch, future_batch, max_context_length=15360, target_columns=None):
            if self.enabled:
                past_batch, future_batch = materialize_covariates(task, past_batch, future_batch, target_columns)
            return original(task, imputation, past_batch, future_batch,
                            max_context_length=max_context_length, target_columns=target_columns)
        native.prepare_covariates_for_timesfm3 = prepare

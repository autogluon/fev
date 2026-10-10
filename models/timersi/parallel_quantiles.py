"""Parallel independent quantile specialists, without changing their training."""
from concurrent.futures import ThreadPoolExecutor
import inspect


def parallel_fit_quantiles(original, workers=2):
    signature = inspect.signature(original)

    def fit(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        quantiles = list(bound.arguments['ql'])

        def one(q):
            arguments = {**bound.arguments, 'ql': [q]}
            return original(**arguments)[0]

        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(one, quantiles))
    return fit


class ParallelProgramLoader:
    """Opt-in, process-local hook; independent quantile training keeps order."""
    def __init__(self, workers=2):
        self.workers = workers
        self.enabled = False

    def install(self):
        import program_runner
        original_load = program_runner.load_pipeline

        def load(folder):
            module = original_load(folder)
            solar = getattr(module, 'solar', None)
            original = getattr(solar, 'fit_quantiles', None)
            if self.enabled and original and 'ql' in inspect.signature(original).parameters:
                solar.fit_quantiles = parallel_fit_quantiles(original, self.workers)
            return module
        program_runner.load_pipeline = load

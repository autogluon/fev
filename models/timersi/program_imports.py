"""Keep candidate helper modules local to one program invocation."""

from contextlib import contextmanager
from functools import wraps
import importlib
from pathlib import Path
import sys


@contextmanager
def isolated_core_imports(root):
    root = Path(root).resolve()
    names = {path.stem for path in root.glob("*.py")}
    original_modules = {name: sys.modules[name] for name in names if name in sys.modules}
    original_path = list(sys.path)
    try:
        yield
    finally:
        sys.path[:] = original_path
        for name in names:
            if name in original_modules:
                sys.modules[name] = original_modules[name]
            else:
                module = sys.modules.get(name)
                filename = getattr(module, "__file__", None)
                if filename and Path(filename).resolve().is_relative_to(root / "programs"):
                    del sys.modules[name]


def stabilize_local_model(module):
    """Freeze LightGBM's calculation layout, preserving recipe and seeds."""
    original = module.fit_predict

    @wraps(original)
    def fit(*args, **kwargs):
        kwargs['lgb_params'] = {**(kwargs.get('lgb_params') or {}),
                                'deterministic': True, 'force_col_wise': True}
        return original(*args, **kwargs)

    module.fit_predict = fit


def install(deterministic_local_programs=()):
    import program_runner
    root = Path(program_runner.__file__).resolve().parent
    original = program_runner.ProgramRunner.predict
    selected = set(deterministic_local_programs)
    if selected:
        original_load = program_runner.load_pipeline

        @wraps(original_load)
        def load(folder):
            module = original_load(folder)
            if Path(folder).name in selected:
                local = importlib.import_module('localmodel')
                if Path(local.__file__).resolve().parent != Path(folder).resolve():
                    raise ValueError('Selected local model belongs to a different program')
                stabilize_local_model(local)
            return module

        program_runner.load_pipeline = load

    @wraps(original)
    def predict(*args, **kwargs):
        # Candidate imports remain available throughout preprocess/engineer/
        # postprocess, including lazy imports, then core bindings are restored.
        with isolated_core_imports(root):
            return original(*args, **kwargs)

    program_runner.ProgramRunner.predict = predict

from pathlib import Path
import sys
from fev import ForecastingModel
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data_decode import CovariateDecodeSwitch
from forecast_cache import ForecastCache
from native import NativeModel
from optimized_pipeline import LazyPipeline
from parallel_quantiles import ParallelProgramLoader
from program_imports import install as install_program_isolation
from view_cache import cache_window

class TimeRSIModel(ForecastingModel):
    model_name = 'timersi'
    # This field describes foundation-model pretraining overlap.
    trained_on_datasets = []

    def __init__(self, checkpoint_path='google/timesfm-3.0-pytorch', device=None):
        super().__init__()
        self.native = NativeModel(checkpoint_path=checkpoint_path, device=device, per_core_batch_size=64)
        self._ready = False
        self._cache = None

    def _prepare(self):
        if self._ready:
            return
        import torch
        install_program_isolation(('p9fa56fb7144b7c68',))
        self._parallel = ParallelProgramLoader(workers=2)
        self._parallel.install()
        self._parallel.enabled = True
        self._decode = CovariateDecodeSwitch()
        self._decode.install()
        self._decode.enabled = True
        forecaster = self.native._get_forecaster()
        use_cuda = torch.cuda.is_available() and self.native.device != 'cpu'
        synchronize = torch.cuda.synchronize if use_cuda else lambda: None
        self._cache = ForecastCache(forecaster.predict_batch, synchronize)
        self._cache.enabled = True
        forecaster.predict_batch = self._cache
        self._ready = True

    def predict_windows(self, task, limit=None):
        self._prepare()
        import torch
        runtime = LazyPipeline(task, self.native)
        outputs = []
        with self._record_inference_time():
            for index, window in enumerate(task.iter_windows(num_proc=1)):
                if limit is not None and index >= limit:
                    break
                self._cache.clear_window()
                outputs.append(runtime.predict_window(cache_window(window), index))
            if torch.cuda.is_available() and self.native.device != 'cpu':
                torch.cuda.synchronize()
        return outputs

    def _fit_predict(self, task):
        return self.predict_windows(task)

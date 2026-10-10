"""Window-local exact-input memoization; never reads archived forecasts."""
from collections import OrderedDict
import copy
import hashlib
import pickle
import time


class ForecastCache:
    def __init__(self, predictor, synchronize=lambda: None, max_bytes=512 * 2**20):
        self.predictor, self.synchronize, self.max_bytes = predictor, synchronize, max_bytes
        self.enabled = False
        self.entries = OrderedDict()
        self.bytes = 0
        self.reset_statistics()

    def reset_statistics(self):
        self.calls = self.hits = self.native_calls = 0
        self.native_seconds = 0.

    def clear_window(self):
        self.entries.clear()
        self.bytes = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        key = hashlib.sha256(pickle.dumps((args, kwargs), protocol=5)).digest() if self.enabled else None
        if self.enabled and key in self.entries:
            self.hits += 1
            value, size = self.entries.pop(key)
            self.entries[key] = value, size
            return copy.deepcopy(value)
        self.synchronize()
        started = time.perf_counter()
        try:
            value = list(self.predictor(*args, **kwargs))
        finally:
            self.synchronize()
            self.native_seconds += time.perf_counter() - started
            self.native_calls += 1
        if self.enabled:
            size = len(pickle.dumps(value, protocol=5))
            if size <= self.max_bytes:
                while self.entries and self.bytes + size > self.max_bytes:
                    _, (_, old_size) = self.entries.popitem(last=False)
                    self.bytes -= old_size
                self.entries[key] = copy.deepcopy(value), size
                self.bytes += size
        return value

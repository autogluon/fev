from __future__ import annotations

import inspect
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import TYPE_CHECKING, Type

import datasets
from typing_extensions import Self

if TYPE_CHECKING:
    from .task import Task


class ForecastingModel(ABC):
    """Base class for forecasting model wrappers.

    Concrete subclasses are registered on import and discoverable via `get_model_cls()`.

    How to implement a subclass:
    - Set `model_name` class attribute if the default doesn't work for you.
    - Implement _fit_predict(task) -> predictions for all windows in the task.
    - Each _fit_predict call should be independent (don't carry over state from prior tasks).
    - Caching expensive resources (weights, tokenizers) on self across calls is fine.
    - For pretrained models, set `trained_on_datasets` to the list of HF dataset configs
      (from autogluon/fev_datasets) that were in the model's training data. This is used
      to flag potential data leakage during evaluation.
    """

    _registry: dict[str, type] = {}
    model_name: str | None = None
    trained_on_datasets: list[str] = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not inspect.isabstract(cls):
            if cls.model_name is None:
                cls.model_name = cls.__name__.removesuffix("Model").lower()
            if cls.model_name in cls._registry and cls._registry[cls.model_name] is not cls:
                raise ValueError(
                    f"Model name '{cls.model_name}' is already registered by "
                    f"{cls._registry[cls.model_name].__qualname__}"
                )
            cls._registry[cls.model_name] = cls

    def __init__(self):
        # Set these directly or use _record_training_time / _record_inference_time helpers
        self.training_time: float = 0.0
        self.inference_time: float = 0.0

    def fit_predict(self, task: Task) -> list[datasets.DatasetDict]:
        """Produce predictions for all windows in the task.

        Resets timing attributes before each call. Subclasses implement `_fit_predict`.
        """
        self.training_time = 0.0
        self.inference_time = 0.0
        return self._fit_predict(task)

    @abstractmethod
    def _fit_predict(self, task: Task) -> list[datasets.DatasetDict]:
        """Implement this. Called by `fit_predict` after timing is reset."""
        ...

    @classmethod
    def list_available_models(cls) -> list[str]:
        """Return names of all registered (imported) model subclasses."""
        return sorted(cls._registry.keys())

    @classmethod
    def get_model_cls(cls, model_name: str) -> Type[Self]:
        """Look up a registered model class by name (case-insensitive). Raises ValueError if not found."""
        model_name = model_name.lower()
        if model_name not in cls._registry:
            available = cls.list_available_models()
            raise ValueError(f"Unknown model '{model_name}'. Available: {available}")
        return cls._registry[model_name]

    @contextmanager
    def _record_training_time(self):
        """Optional helper. Accumulates elapsed time into self.training_time."""
        start = time.monotonic()
        try:
            yield
        finally:
            self.training_time += time.monotonic() - start

    @contextmanager
    def _record_inference_time(self):
        """Optional helper. Accumulates elapsed time into self.inference_time."""
        start = time.monotonic()
        try:
            yield
        finally:
            self.inference_time += time.monotonic() - start

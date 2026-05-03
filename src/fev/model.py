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

    Subclasses are registered on import: importing a module that defines a concrete subclass
    is sufficient for it to appear in `list_available_models()` / `get_model_cls()`.
    Registry key is the class name with "Model" suffix stripped, lowercased.
    """

    _registry: dict[str, type] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not inspect.isabstract(cls):
            name = cls.__name__.removesuffix("Model").lower()
            cls._registry[name] = cls

    def __init__(self):
        self.training_time: float = 0.0
        self.inference_time: float = 0.0

    @abstractmethod
    def fit_predict(self, task: Task) -> list[datasets.DatasetDict]:
        """Must be implemented by all models. Produce predictions for all windows in the task."""
        ...

    @classmethod
    def list_available_models(cls) -> list[str]:
        """Return names of all registered (imported) model subclasses."""
        return sorted(cls._registry.keys())

    @classmethod
    def get_model_cls(cls, model_name: str) -> Type[Self]:
        """Look up a registered model class by name. Raises ValueError if not found."""
        model_name = model_name.lower()
        if model_name not in cls._registry:
            available = cls.list_available_models()
            raise ValueError(f"Unknown model '{model_name}'. Available: {available}")
        return cls._registry[model_name]

    @contextmanager
    def _record_training_time(self):
        """Context manager that accumulates elapsed time into self.training_time."""
        start = time.monotonic()
        try:
            yield
        finally:
            self.training_time += time.monotonic() - start

    @contextmanager
    def _record_inference_time(self):
        """Context manager that accumulates elapsed time into self.inference_time."""
        start = time.monotonic()
        try:
            yield
        finally:
            self.inference_time += time.monotonic() - start

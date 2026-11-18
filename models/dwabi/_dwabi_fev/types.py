"""Immutable provider inputs and marginal quantiles."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import pairwise
from types import MappingProxyType

import numpy as np


def _vector_mapping(columns, size, role):
    result = {}
    for name, values in columns.items():
        x = np.array(values, copy=True)
        if x.dtype.kind in "SU":
            x = np.array(values, dtype=object, copy=True)
        if not isinstance(name, str) or not name or x.shape != (size,):
            raise ValueError(f"{role} covariates require nonempty names and {size} values")
        x.setflags(write=False)
        result[name] = x
    return MappingProxyType(result)


@dataclass(frozen=True)
class SeriesPrefix:
    """One item at issuance, with target rows and timestamped covariate roles.

    History has shape (targets, time). NaN denotes an unknown target.
    Past covariates span history only. Known covariates span history and
    the forecast horizon. Supply only values actually available at issue.
    """

    key: str
    history: np.ndarray
    history_dates: np.ndarray
    future_dates: np.ndarray
    past: Mapping = field(default_factory=dict)
    known: Mapping = field(default_factory=dict)
    static: Mapping = field(default_factory=dict)

    def __post_init__(self):
        if not all(isinstance(x, Mapping) for x in (self.past, self.known, self.static)):
            raise ValueError("covariates must be supplied as mappings")
        history = np.array(self.history, dtype=float, copy=True)
        past_dates = np.array(self.history_dates, dtype="datetime64[ns]", copy=True)
        future_dates = np.array(self.future_dates, dtype="datetime64[ns]", copy=True)
        if not isinstance(self.key, str) or not self.key:
            raise ValueError("series key must be a nonempty string")
        if history.ndim != 2 or not all(history.shape) or np.isinf(history).any():
            raise ValueError("history requires nonempty finite or missing target rows")
        if past_dates.shape != history.shape[1:] or future_dates.ndim != 1 or not len(future_dates):
            raise ValueError("timestamps must match the history and a positive horizon")
        clock = np.r_[past_dates, future_dates]
        if np.isnat(clock).any() or (clock[1:] <= clock[:-1]).any():
            raise ValueError("ordered future timestamps must be strictly after history")
        past = _vector_mapping(self.past, len(past_dates), "past")
        known = _vector_mapping(self.known, len(clock), "known")
        if not all(isinstance(name, str) and name for name in self.static):
            raise ValueError("static covariates require nonempty names")
        if any(set(a) & set(b) for a, b in ((past, known), (past, self.static), (known, self.static))):
            raise ValueError("covariate names must occupy one role")
        for array in (history, past_dates, future_dates):
            array.setflags(write=False)
        object.__setattr__(self, "history", history)
        object.__setattr__(self, "history_dates", past_dates)
        object.__setattr__(self, "future_dates", future_dates)
        object.__setattr__(self, "past", past)
        object.__setattr__(self, "known", known)
        object.__setattr__(self, "static", MappingProxyType(dict(self.static)))



@dataclass(frozen=True)
class Forecast:
    """Marginal quantiles on the last axis and their median point decision."""

    values: np.ndarray
    levels: tuple[float, ...]

    def __post_init__(self):
        q = np.array(self.values, dtype=np.float64, copy=True)
        tau = tuple(self.levels)
        if (
            q.ndim < 2
            or q.size == 0
            or q.shape[-1] != len(tau)
            or 0.5 not in tau
            or not all(0 < x < 1 for x in tau)
            or any(a >= b for a, b in pairwise(tau))
        ):
            raise ValueError(
                "quantiles require increasing probability levels including 0.5"
            )
        if not np.isfinite(q).all() or (np.diff(q, axis=-1) < 0).any():
            raise ValueError("forecast quantiles must be finite and ordered")
        q.setflags(write=False)
        object.__setattr__(self, "values", q)
        object.__setattr__(self, "levels", tau)

    @property
    def point(self) -> np.ndarray:
        """The projected median, on the same series and lead axes."""
        return self.values[..., self.levels.index(0.5)]

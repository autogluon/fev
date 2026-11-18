"""Bind the generic point policy to FEV's native weights and issued inputs."""

from pathlib import Path

import numpy as np
ROOT = Path(__file__).resolve().parents[1]


def epsilon(task):
    return float(next(metric.epsilon for metric in task.metrics if metric.name == "WAPE"))


def record(learner, task, family, row, actual, laws):
    """Make the task's single native WAPE weight for each observed cell."""
    valid = np.isfinite(actual)
    count = valid.sum(axis=(0, 1))
    volume = np.maximum(np.where(valid, np.abs(actual), 0.).sum(axis=(0, 1)), count * epsilon(task))
    clocks = {"origin": row["origin"], "issued_at": row["issue_at_ns"],
              "first_target_at": row["first_target_at_ns"], "available_at": row["available_at_ns"]}
    if (volume <= 0.).any():
        learner.observe_scores(task.task_name, family, **clocks, ordinal_cost=None, ordinal_constant=None,
                               menu_wape=None)
        return False
    mass = np.divide(1., len(volume) * volume) * valid
    points = np.stack([laws[member][..., 0] for member in learner.models], axis=-1)
    learner.observe(task.task_name, family, **clocks, points=points, actual=actual, case_weights=mass, include_menu=False)
    return True

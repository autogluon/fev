"""Store and reuse verified recipient-aligned, outcome-free earlier forecasts."""

import json
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np


def array_hash(values):
    return sha256(np.ascontiguousarray(values)).hexdigest()


class ForecastBank:
    def __init__(self, root, inputs):
        self.root = Path(root)
        self.inputs = inputs

    def forecast(self, model, task, row, case, ends, members):
        laws = {}
        for name in members:
            path = self.root / name / task.task_name / f"{row['origin']}.npz"
            expected = {
                "inputs": self.inputs, "member": name, "task": task.task_name, "issue": row,
                "cutoffs_sha256": array_hash(ends.astype("<i8")),
                "lead_times_sha256": array_hash(case[4].astype("<i8")),
                "known": list(task.known_dynamic_columns), "past_only": list(task.past_dynamic_columns),
                "static": list(task.static_columns), "deciles": list(task.quantile_levels),
            }
            if not path.exists():
                law = model._forecast(task, case, ends, (name,))[name]
                metadata = expected | {
                    "law_sha256": array_hash(law),
                    "producer": {"device": model.device, "torch": version("torch"), "dwabi": version("dwabi")},
                }
                path.parent.mkdir(parents=True, exist_ok=True)
                temp = None
                try:
                    with NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
                        temp = Path(stream.name)
                        np.savez_compressed(stream, law=law, metadata=json.dumps(metadata, allow_nan=False))
                    try:
                        path.hardlink_to(temp)
                    except FileExistsError:
                        pass  # Reuse and verify the first complete writer's receipt.
                finally:
                    if temp is not None:
                        temp.unlink()
            with np.load(path, allow_pickle=False) as saved:
                law = saved["law"]
                metadata = json.loads(str(saved["metadata"].item()))
            producer = metadata.pop("producer")
            if (producer["dwabi"] != version("dwabi") or not producer["device"] or not producer["torch"]
                    or metadata.pop("law_sha256") != array_hash(law) or metadata != expected):
                raise ValueError(f"{path}: earlier law, producer or issue-time binding changed.")
            shape = (*case[4].shape, len(task.target_columns), len(task.quantile_levels) + 1)
            if law.shape != shape or not np.isfinite(law).all():
                raise ValueError(f"{path}: the forecast does not cover the complete native recipient axes.")
            laws[name] = law
        return laws

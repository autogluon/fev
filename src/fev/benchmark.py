from pathlib import Path

import requests
import yaml

from .task import Task


class Benchmark:
    """A time series forecasting benchmark consisting of multiple [`Task`][fev.Task] objects.

    Attributes
    ----------
    tasks : list[Task]
        Collection of tasks in the benchmark.
    """

    def __init__(self, tasks: list[Task]):
        for t in tasks:
            if not isinstance(t, Task):
                raise ValueError(f"`tasks` must be a list of `Task` objects (got {type(t)})")
        self.tasks: list[Task] = tasks  # declare type explicitly to correctly show up in the docs

    @classmethod
    def from_yaml(cls, file_path: str | Path) -> "Benchmark":
        """Load benchmark definition from a YAML file.

        The YAML file should contain the key `'tasks'` with a list of dictionaries with task definitions.

            tasks:
            - dataset_path: autogluon/chronos_datasets
              dataset_config: m4_hourly
              horizon: 24
              num_windows: 2
            - dataset_path: autogluon/chronos_datasets
              dataset_config: monash_cif_2016
              horizon: 12

        Parameters
        ----------
        file_path : str | Path
            URL or path of a YAML file containing the task definitions.
        """
        try:
            if str(file_path).startswith(("http://", "https://")):
                response = requests.get(file_path)
                response.raise_for_status()
                config = yaml.safe_load(response.text)
            else:
                with open(file_path) as file:
                    config = yaml.safe_load(file)
        except Exception:
            raise ValueError("Failed to load the file")

        return cls.from_list(config["tasks"])

    @classmethod
    def from_list(cls, task_configs: list[dict]) -> "Benchmark":
        """Load benchmark definition from a list of dictionaries.

        Each dictionary must follow the schema compatible with a `fev.task.Task`.
        """
        return cls(tasks=[Task(**conf) for conf in task_configs])

from pathlib import Path

import requests
import yaml

from .task import Task, TaskGenerator


class Benchmark:
    """Benchmark consisting of multiple tasks.

    Attributes
    ----------
    tasks : list[Task]
        Collection of tasks in the benchmark.
    """

    def __init__(self, tasks: list[Task]):
        self.tasks = list(tasks)
        for task in self.tasks:
            assert isinstance(task, Task), "`tasks` must be a list of list of `Task` objects"

    @classmethod
    def from_yaml(cls, file_path: str | Path) -> "Benchmark":
        """Load benchmark definition from a YAML file.

        The YAML file should contain the key 'tasks' with a list of values with task definitions.

            tasks:
            - dataset_path: autogluon/chronos_datasets
              dataset_config: m4_hourly
              horizon: 24
            - dataset_path: autogluon/chronos_datasets
              dataset_config: monash_cif_2016
              horizon: 12

        It is possible to create multiple variants of each task using the `variants` key. For example, the following
        YAML config will generate 3 tasks corresponding to a 3-window backtest:

            tasks:
            - dataset_path: autogluon/chronos_datasets
              dataset_config: m4_hourly
              horizon: 24
              variants:
                - cutoff: -64
                - cutoff: -48
                - cutoff: -24

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

        Each dictionary must follow the schema compatible with a `fev.task.TaskGenerator`.
        """
        tasks = []
        for conf in task_configs:
            tasks.extend(TaskGenerator(**conf).generate_tasks())
        return cls(tasks=tasks)

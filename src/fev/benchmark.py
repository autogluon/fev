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
    task_generators : list[TaskGenerator]
        Collection of task generators used to create individual tasks in the benchmark.
    """

    def __init__(self, tasks: list[Task | TaskGenerator]):
        self.task_generators = []
        for t in tasks:
            if isinstance(t, TaskGenerator):
                self.task_generators.append(t)
            elif isinstance(t, Task):
                self.task_generators.append(TaskGenerator(**t.to_dict()))
            else:
                raise ValueError(f"`tasks` must be a list of `Task` or `TaskGenerator` objects (got {type(t)})")

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
        return cls(tasks=[TaskGenerator(**conf) for conf in task_configs])

    @property
    def tasks(self) -> list[Task]:
        """List of tasks in the benchmark."""
        tasks = []
        for task_gen in self.task_generators:
            tasks.extend(task_gen.generate_tasks())
        return tasks

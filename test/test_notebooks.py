from datetime import datetime
from pathlib import Path

import nbformat
import pytest
from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor

notebooks_dir = Path(__file__).parent.parent / "docs" / "tutorials"
notebook_files = [
    notebooks_dir / nb_file
    for nb_file in [
        "01-quickstart.ipynb",
        "02-dataset-format.ipynb",
        "03-tasks-and-benchmarks.ipynb",
        "04-adapters.ipynb",
        "05-add-your-model.ipynb",
    ]
]


def _drop_pip_installs(source: str) -> str:
    return "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith(("!pip install", "%pip install"))
    )


def _print_slowest_cells(nb, capsys, n=3):
    """Print the slowest cells, using the timings nbclient records in each cell's metadata."""
    timings = []
    for i, cell in enumerate(nb.cells):
        execution = cell.get("metadata", {}).get("execution", {})
        if "iopub.execute_input" in execution and "shell.execute_reply" in execution:
            start = datetime.fromisoformat(execution["iopub.execute_input"].replace("Z", "+00:00"))
            end = datetime.fromisoformat(execution["shell.execute_reply"].replace("Z", "+00:00"))
            timings.append(((end - start).total_seconds(), i, cell.source.splitlines()[0] if cell.source else ""))
    with capsys.disabled():
        for seconds, i, first_line in sorted(timings, reverse=True)[:n]:
            print(f"\n  {seconds:6.1f}s  cell {i}: {first_line[:80]}", end="")


@pytest.mark.parametrize("notebook_path", notebook_files)
def test_notebooks_run_without_errors(notebook_path, capsys):
    with notebook_path.open("r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    # Notebook dependencies come from the `test` extras. Installing them again in CI is slow and may change
    # package versions in the environment shared with the other tests.
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.source = _drop_pip_installs(cell.source)

    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")

    try:
        ep.preprocess(nb, {"metadata": {"path": notebooks_dir}})
        _print_slowest_cells(nb, capsys)
        print(f"{notebook_path.name} ran successfully.")
    except CellExecutionError:
        pytest.fail(f"Error executing the notebook {notebook_path.name}. See the traceback above.")
    except Exception as e:
        pytest.fail(f"An error occurred while running the notebook {notebook_path.name}: {e}")

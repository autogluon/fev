from pathlib import Path

import nbformat
import pytest
from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor

notebooks_dir = Path(__file__).parent.parent / "docs"
notebook_files = [
    notebooks_dir / nb_file
    for nb_file in [
        "01-quickstart.ipynb",
        "02-dataset-format.ipynb",
        "03-tasks-and-benchmarks.ipynb",
        "04-models.ipynb",
    ]
]


@pytest.mark.parametrize("notebook_path", notebook_files)
def test_notebooks_run_without_errors(notebook_path):
    with notebook_path.open("r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")

    try:
        ep.preprocess(nb, {"metadata": {"path": notebooks_dir}})
        print(f"{notebook_path.name} ran successfully.")
    except CellExecutionError:
        pytest.fail(f"Error executing the notebook {notebook_path.name}. See the traceback above.")
    except Exception as e:
        pytest.fail(f"An error occurred while running the notebook {notebook_path.name}: {e}")

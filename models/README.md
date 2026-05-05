# Model Wrappers

Each subfolder contains a forecasting model wrapper. Below is how to run evaluation for existing wrappers and how to add your own.

## Prerequisites

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repo and install fev in editable mode
git clone https://github.com/autogluon/fev.git
cd fev
uv pip install -e .
```

## Evaluate a model

```bash
python models/evaluate.py -m chronos
```

Options:
- `-m` — model name (must match a subfolder in `models/`)
- `-b` — path or URL to benchmark YAML (default: `fev_bench_mini`)
- `-n` — display name for results (default: same as `-m`)
- `-k` — JSON dict of kwargs passed to the model constructor
- `-t` — limit number of tasks (useful for quick testing)

Model dependencies are installed automatically in an ephemeral environment. Your project environment is not modified.

## Add a custom model

1. Create a folder `models/<name>/` where `<name>` is how you'll refer to the model with `-m`.

2. Add `model.py` with a subclass of `ForecastingModel`. Set `model_name` to match the folder name.

```python
import datasets
import fev
from fev.model import ForecastingModel

class MyModel(ForecastingModel):
    model_name = "my-model"  # must match the folder name

    # List HF dataset configs (from autogluon/fev_datasets) used during pretraining.
    # This is used to flag potential data leakage during evaluation.
    trained_on_datasets = ["kdd_cup_2022_10T", "m5_1D"]

    def __init__(self, model_size: str = "small"):
        super().__init__()
        self.model_size = model_size

    def _fit_predict(self, task: fev.Task) -> list[datasets.DatasetDict]:
        ...
```

3. Add `requirements.txt` with pinned dependencies for the model.

4. For pretrained models, set `trained_on_datasets` to the list of dataset configs from `autogluon/fev_datasets` that overlap with the model's training data. Leave empty for models that train from scratch.

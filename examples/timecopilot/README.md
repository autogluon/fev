# TimeCopilot `fev` Experiments

This project demonstrates the evaluation of a foundation model ensemble built using the [TimeCopilot](https://timecopilot.dev) library on the [fev](https://github.com/autogluon/fev/) benchmark.

TimeCopilot is an open‑source AI agent for time series forecasting that provides a unified interface to multiple forecasting approaches, from foundation models to classical statistical, machine learning, and deep learning methods, along with built‑in ensemble capabilities for robust and explainable forecasting.

## Model Description

This ensemble leverages [**TimeCopilot's MedianEnsemble**](https://timecopilot.dev/api/models/ensembles/#timecopilot.models.ensembles.median.MedianEnsemble) feature, which combines two state-of-the-art foundation models:

- [**TiRex** (NX-AI)](https://timecopilot.dev/api/models/foundation/models/#timecopilot.models.foundation.tirex.TiRex)
- [**Chronos** (AWS AI Labs)](https://timecopilot.dev/api/models/foundation/models/#timecopilot.models.foundation.chronos.Chronos)

## Setup

### Prerequisites
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- AWS CLI configured (for distributed evaluation)
- [Modal](https://modal.com/) account (for distributed evaluation)

### Installation

```bash
# Install dependencies
uv sync
```

## Evaluation Methods

### 1. Local Evaluation

Run evaluation sequentially (locally):

```bash
uv run -m src.evaluate_model --num-tasks 2
```

Remove `--num-tasks` parameter to run on all tasks. Results are saved to `timecopilot.csv` in `fev` format.

### 2. Distributed Evaluation (Recommended)

#### 2.1 Evaluate ensemble

Evaluate all dataset configurations in parallel using [modal](https://modal.com/):

```bash
# Run distributed evaluation on Modal cloud
uv run modal run --detach -m src.evaluate_model_modal
```

This creates one GPU job per dataset configuration, significantly reducing evaluation time.

**Infrastructure:**
- **GPU**: A10G per job
- **CPU**: 8 cores per job  
- **Timeout**: 3 hours per job
- **Storage**: S3 bucket for data and results

#### 2.2 Collect Results

Download and consolidate results from distributed evaluation:

```bash
# Download all results from S3 and create consolidated CSV
uv run python -m src.download_results
```

Results are saved to `timecopilot.csv` in `fev` format.

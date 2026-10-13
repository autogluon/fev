# t0-gamma

`t0-gamma` is an open-weights time-series forecasting model from [The Forecasting Company](https://theforecastingcompany.com/) and an upgrade to `t0-beta`.

- Weights and model card: https://huggingface.co/theforecastingcompany/t0-gamma
- Runtime: [`tfc-t0`](https://pypi.org/project/tfc-t0/) 0.6.0 (PyTorch)

This submission adds a new wrapper and results for all 100 fev-bench tasks.

```bash
uv run python models/evaluate.py -m t0-gamma -b benchmarks/fev_bench/tasks.yaml
```

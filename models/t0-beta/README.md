# t0-beta

`t0-beta` is a 256M-parameter open-weights time series foundation model from
[The Forecasting Company](https://theforecastingcompany.com/). It is a
decoder-style patch transformer that emits 21 quantile levels natively and
conditions on past and known-future covariates.

- Weights: https://huggingface.co/theforecastingcompany/t0-beta
- Code: https://github.com/theforecastingcompany/tfc-t0
- Runtime: [`tfc-t0`](https://pypi.org/project/tfc-t0/) `>=0.5.0`

```bash
python models/evaluate.py -m t0-beta
```

## How the wrapper maps a task

A task's columns become variates of one t0 sample, which attend to one another
through the model's group attention:

| Task column | t0 variate |
| --- | --- |
| every target column | `TARGET`, all forecast in one pass |
| `known_dynamic_columns` | `FUTURE`, spanning context and horizon |
| `past_dynamic_columns` | `HISTORICAL`, stopping at the forecast start |

Non-numeric covariates are label-encoded per request, as the model has no
categorical embedding. Context is truncated to the most recent 8192 steps.

Ablations: `-k '{"use_covariates": false}'` drops the covariate rows, and
`-k '{"as_univariate": true}'` forecasts each target column on its own.

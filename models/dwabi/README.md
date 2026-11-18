# D-wabi on FEV-Bench

D-wabi combines forecasts from fixed pretrained models. Each FEV task learns its combination from its own earlier forecast errors. The same procedure runs on every task. Its outputs are one point and nine ordered quantiles for every target, recipient, lead and official window. The combination is task-trained and the component checkpoints are frozen.

## What each model sees

FEV constructs the forecast windows. Each provider receives the history and future inputs returned by `EvaluationWindow.get_input_data()`. Declared known-future values span the forecast horizon. Past-only covariates stop at the recipient's cutoff. Static covariates retain their declared role. Providers can use the fields their architectures support. For example, the stock Toto-2 adapter consumes target history. Where recipient cutoffs differ within one FEV window, D-wabi groups recipients by their actual cutoff for inference and restores FEV's original row order. FEV determines the cadence, missing observations, minimum and maximum context, horizon, quantile levels and scoring settings. [FEV's forecast-time data contract](https://github.com/autogluon/fev/blob/844509e5211292620fbd9d5f24695f1d27da767a/src/fev/task.py#L60-L111)

Chronos-2 target-encodes categorical covariates when forecasting a single target. If that recipient's entire target history is missing, its encoder has no response to fit. For this provider only, D-wabi marks the unencodable categories as missing and passes the numeric drivers through. The fallback applied to one recipient during calibration and zero recipients in the official forecast windows. [Chronos-2 input handling](_dwabi_fev/chronos_missing.py)

## How it fits the combination

The base roster comprises t0-gamma, ci-proto, Timestral-209M, TiRex-2, TimesFM-3, Chronos-2 and Toto-2.0-1B. TimeH2O-v1 is eligible for the quantile pool on cohorts that fit its published input budget. Model identifiers, checkpoint revisions, file digests and package versions appear in [provider_pins.json](provider_pins.json).

1. The calibration calendar uses timestamps, recipient membership and source lineage. It places earlier forecasting episodes inside the history FEV supplies at each recipient's first official cutoff. A task's `max_context_length` therefore limits both the evidence used for fitting and the history a provider sees. The first set ends before the earliest official issue in its physical source family. If that provides fewer than three disjoint comparisons, the scheduler also searches the same task's later eligible history within this prefix, protecting sibling tasks' official target intervals. Family identifiers only set these time exclusions. Target values from other tasks do not enter a task's fit.
2. For the point, seven member forecasts are sorted at each recipient, lead and target. The package learns ordinal weights on the original members by minimizing the task's WAPE on mature, disjoint calibration forecasts. With more than three observations it minimizes worst leave-one-origin-out excess loss versus the equal-weight median. With insufficient evidence it keeps equal weights. The seven members' deciles estimate how much a cell contributes to the WAPE denominator. The package uses that estimate and the target's observed support to compute its final point. [Point solver](https://github.com/AbishekSriramulu/dwabi/blob/v0.1.1.post2/src/dwabi/point_policy.py) · [WAPE point adjustment](https://github.com/AbishekSriramulu/dwabi/blob/v0.1.1.post2/src/dwabi/wape.py)
3. For the quantiles, the fitter ranks individual models by FEV's native scaled quantile loss (SQL) on common, mature, disjoint earlier windows of this task. Three complete comparisons admit the three best members. Before that point it uses all seven base members. It orders each selected member's deciles and takes a coordinatewise median. The optional eight-member comparison uses the same qualification rule and a budget of two million input cells. Cell counts use up to 16,384 history steps, the full requested horizon, the number of targets and the number of declared known-future covariates. The selected three can include TimeH2O.
4. When only task-local windows provide enough support, the seven-member fallback shortlist must beat the full-seven median on four earlier, actually issued validation forecasts. It must improve average native SQL and maintain its pairwise win rate against the individual members. Each held forecast is selected using outcomes strictly before that forecast's issue. All four outcomes must have arrived before using their validation decision. Any missing required evidence leaves the ordinary fallback in place.

All 100 submitted tasks leave `max_context_length` unset. Their full pre-cutoff histories are available for selecting earlier calibration windows, and each provider uses its supported context length. The fitter freezes a task's learned choices before its first official issue. On later issues it receives the advancing native history, recomputes input-size eligibility, and issues forecasts with those frozen choices. This benchmark configuration estimates weights and qualifies shortlists independently for each task. [FEV protocol, section 7.1](https://arxiv.org/html/2509.26468v1#S7.SS1)

## Run from source

Use Linux, Python 3.12 and a CUDA-capable NVIDIA GPU. Request access to the gated TimeH2O checkpoint from H2O. The pinned packages have conflicting upstream dependency declarations. Install the exact lock with `--no-deps`. The separate PyTorch CUDA wheel and CUDA toolkit paths below reproduce the tested environment. [Stock evaluator](../../models/evaluate.py)

Set `PYTHONHASHSEED=0` in the shell before starting Python for both calibration and evaluation. The pinned TimesFM-3 adapter converts text categories to numeric buckets with Python's process-wide string hash. The submitted forecasts used seed zero. In a notebook, start the kernel with that environment setting. [TimesFM-3 categorical encoding](../timesfm-3/model.py)

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
export PYTHONHASHSEED=0
uv pip install --no-deps --index-url https://download.pytorch.org/whl/cu128 'torch==2.10.0+cu128'
uv pip install --no-deps -r models/dwabi/requirements.txt
uv pip install --no-deps -e .
export CUDA_HOME="$(python -c 'import sysconfig; print(sysconfig.get_path("purelib") + "/nvidia/cu13")')"
export CUDA_PATH="$CUDA_HOME"
python models/dwabi/prepare_data.py ./fev-data
export FEV_DATASETS_PREFIX="$PWD/fev-data"
python models/evaluate.py -m dwabi -d -b benchmarks/fev_bench/tasks.yaml -n D-wabi \
    -k '{"method":"native_cutoff"}'
```

The normal evaluator call builds the input calendar and fits each task on demand. Preparation can also run separately to resume earlier forecasts or allocate one provider per GPU job. The optional forecast bank binds each forecast to its provider, task definition, recipient order, cutoff and the values and missingness in FEV's past and known-future inputs. The final preparation call verifies the files and writes the decisions to a fresh policy file. Use a fresh bank directory when inputs change or an older receipt lacks these checks. Rebuild older prepared policies for tasks that specify a history cap.

```bash
export PYTHONHASHSEED=0
for member in t0 ci timestral tirex timesfm3 chronos2 toto2 timeh2o; do
    python models/dwabi/prepare_cutoff.py benchmarks/fev_bench/tasks.yaml \
        --forecast-bank ./dwabi-earlier --issue-member "$member"
done
python models/dwabi/prepare_cutoff.py benchmarks/fev_bench/tasks.yaml ./dwabi-policy.json \
    --forecast-bank ./dwabi-earlier
export DWABI_FEV_POLICY="$PWD/dwabi-policy.json"
python models/evaluate.py -m dwabi -d -b benchmarks/fev_bench/tasks.yaml -n D-wabi \
    -k '{"method":"native_cutoff"}'
```

For already-downloaded checkpoints, give `--checkpoint-root` to preparation and `checkpoint_root` in the evaluator's model arguments. Under that directory each provider key contains the files named in its pin. When using a prepared policy, report preparation time separately from the scorer's per-task inference time. Stock `Task.evaluation_summary()` scores the issued forecasts. Package 0.1.1.post1 is pinned for research reproduction. Its seven forecasting modules are byte-identical to 0.1.1.post2. [Package release](https://pypi.org/project/dwabi/0.1.1.post2/)

With pytest installed in the pinned environment, run `python -m pytest models/dwabi` for the adapter's CPU regression tests.

## Checkpoint attribution

Uses ci-proto, released by Continuous Intelligence. Continuous Intelligence does not endorse or approve this product or service. Checkpoint use is subject to its [original license](CI_PROTO_LICENSE.txt). Each other provider retains its published terms, including the [Timestral license](https://huggingface.co/mistralai/Timestral-209M/blob/9874d252ab0aae16f7f7c999f98f98d6ad7d8365/LICENSE) and the [TimeH2O noncommercial license](https://huggingface.co/h2oai/timeh2o-v1/tree/1310be782553943a4c520d2810a98ce8f56b2709). The pretraining-overlap metadata follows the declarations for the pinned checkpoints in the FEV checkout. This repository distributes the adapter code and checkpoint references.

"""Native forecast adapters for t0-gamma, Timestral and CI-Proto."""

from collections import defaultdict
from pathlib import Path

import numpy as np

from .types import Forecast

DECILES = tuple(i / 10 for i in range(1, 10))


def _numeric(values):
    try:
        return np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError):
        vocabulary = sorted({str(x) for x in values if not _missing(x)})
        return _codes(values, vocabulary)


def _missing(value):
    return value is None or (isinstance(value, (float, np.floating)) and np.isnan(value))


def _codes(values, vocabulary):
    indexes = {category: i for i, category in enumerate(vocabulary)}
    return np.asarray([np.nan if _missing(x) else indexes[str(x)] for x in values], dtype=np.float32)


def _categories(series, *, numeric_text=True):
    present = [x for values in series for x in values if not _missing(x)]
    if present and all(isinstance(x, (bool, np.bool_)) for x in present):
        return 'bool', ()
    if numeric_text or all(not isinstance(x, (str, bytes)) for x in present):
        try:
            numeric = np.asarray(present, dtype=np.float64)
        except (ValueError, TypeError):
            pass
        else:
            if not np.isnan(numeric).any():
                return 'numeric', ()
    return 'text', tuple(sorted({str(x) for x in present}))


def _pad(arrays, width, fill=np.nan):
    shape = (len(arrays), *arrays[0].shape[:-1], width)
    out = np.full(shape, fill, dtype=arrays[0].dtype)
    for i, value in enumerate(arrays):
        out[i, ..., width - value.shape[-1]:] = value
    return out


def _deciles(values, levels):
    indexes = [int(np.argmin(abs(np.asarray(levels) - level))) for level in DECILES]
    if not np.allclose(np.asarray(levels)[indexes], DECILES, rtol=0, atol=1e-6):
        raise ValueError('Checkpoint must provide the nine requested deciles.')
    return Forecast(np.asarray(values)[..., indexes], DECILES)


def _ci_bound_batches():
    import torch
    from continuousintelligence.ci_proto import _forecast

    original = _forecast._packed_predict
    if getattr(original, '_forephase_bounded', False):
        return

    def predict(model, cases, horizon, tta):
        signs = 2 if tta['postprocess'] and tta['tta_sign_flip'] else 1
        rows = len(cases) * len(cases[0].context) * signs
        width = max(x.context.shape[-1] for x in cases) + max(horizon, model.config.future_len)
        if len(cases) == 1 or (rows <= 512 and rows * width <= 2**20):
            try:
                return original(model, cases, horizon, tta)
            except torch.OutOfMemoryError:
                if len(cases) == 1:
                    raise
            torch.cuda.empty_cache()
        mid = len(cases) // 2
        return torch.cat((predict(model, cases[:mid], horizon, tta),
                          predict(model, cases[mid:], horizon, tta)))

    predict._forephase_bounded = True
    _forecast._packed_predict = predict


class T0Member:
    """t0-gamma with within-series variate attention and local native A2."""

    def __init__(self, checkpoint, device):
        import t0
        import torch

        self.sdk, self.torch = t0, torch
        self.device = device
        self.model = t0.T0Forecaster.from_pretrained(checkpoint, local_files_only=True).eval().to(device)

    def _issue(self, row, known_names, past_names):
        torch, sdk = self.torch, self.sdk
        history = row.history[:, -8192:].astype(np.float32)
        length, horizon = history.shape[-1], row.future_dates.size
        known = [_numeric(row.known[name])[-length - horizon:] for name in known_names]
        x = sdk.TimeSeries.from_array(torch.from_numpy(history)[None], horizon=horizon,
                                      future_covariates=torch.from_numpy(np.stack(known))[None] if known else None)
        if not row.past:
            return x
        past = np.stack([_numeric(row.past[name])[-length:] for name in past_names])
        shape = len(past), length + horizon
        observed = torch.from_numpy(np.isfinite(past))
        values = torch.zeros(shape, dtype=torch.float32)
        values[:, :length] = torch.from_numpy(np.where(np.isfinite(past), past, 0.0))
        mask = torch.full(shape, sdk.MaskType.MISSING, dtype=torch.int8)
        mask[:, :length] = torch.where(observed, sdk.MaskType.VALID, sdk.MaskType.MISSING)
        return sdk.TimeSeries(torch.cat((x.variates, values)), torch.cat((x.mask, mask)),
                              torch.cat((x.group_ids, torch.zeros(shape, dtype=torch.long))),
                              torch.cat((x.variate_type, torch.full(shape, sdk.VariateType.HISTORICAL, dtype=torch.long))))

    def __call__(self, panel):
        t0, torch = self.sdk, self.torch
        horizon, targets = panel[0].future_dates.size, panel[0].history.shape[0]
        past_names, known_names = tuple(panel[0].past), tuple(panel[0].known)
        count = targets + len(past_names) + len(known_names)
        out = np.empty((len(panel), horizon, targets, len(DECILES)))
        groups = defaultdict(list)
        for i, row in enumerate(panel):
            context = min(row.history.shape[-1], 8192) if row.past or row.known else 0
            groups[context].append(i)
        batch = max(1, 32 // count)
        for members in groups.values():
            for offset in range(0, len(members), batch):
                indexes = members[offset:offset + batch]
                rows = [self._issue(panel[i], known_names, past_names) for i in indexes]
                packed = t0.TimeSeries.batch(rows).to(self.device)
                with torch.inference_mode():
                    q = self.model.predict(packed, horizon=horizon, quantile_levels=list(DECILES),
                                           context_length=packed.seq_len - horizon, symmetric_averaging=True,
                                           trim_leading_missing=True, clip_nonnegative=True, snap_integers=True)
                out[indexes] = q.quantiles.float().cpu().numpy().reshape(
                    len(indexes), targets, horizon, len(DECILES)
                ).transpose(0, 2, 1, 3)
        return Forecast(out, DECILES)


class TimestralMember:
    """Timestral with issued-window covariate types and bounded batches."""

    def __init__(self, checkpoint, device):
        from timestral import TimestralForecaster

        self.model = TimestralForecaster.from_pretrained(checkpoint, device=device, local_files_only=True)
        if self.model.config.split_context_horizon_attn:
            from ._timestral_runtime import BoundedTimestral

            self.model.model = BoundedTimestral(self.model.model)

    def _batch(self, panel, kinds, horizon, length):
        targets = [row.history[:, -length:].astype(np.float32) for row in panel]
        args = {}
        for role, tail in [('past', 0), ('known', horizon)]:
            for categorical in (False, True):
                names = [name for name in getattr(panel[0], role) if (kinds[name][0] != 'numeric') == categorical]
                key = f'{"categorical_" if categorical else ""}{role}_covariates'
                if names:
                    matrices = []
                    for row in panel:
                        values = getattr(row, role)
                        columns = []
                        for name in names:
                            kind, code = kinds[name]
                            column = values[name][-length - tail:]
                            columns.append(_codes(column, code) if kind == 'text' else _numeric(column))
                        matrices.append(np.stack(columns))
                    args[key] = _pad(matrices, length + tail)
        return self.model.forecast(_pad(targets, length), horizon, **args).quantiles.double().cpu().numpy()

    def __call__(self, panel):
        import torch

        horizon = panel[0].future_dates.size
        targets = panel[0].history.shape[0]
        count = targets + len(panel[0].past) + len(panel[0].known)
        kind = {name: _categories([getattr(row, role)[name] for row in panel])
                for role in ('past', 'known') for name in getattr(panel[0], role)}
        lengths = [min(x.history.shape[-1], self.model.max_context) for x in panel]
        out = np.empty((len(panel), horizon, targets, len(self.model.quantile_levels)))

        def run(indexes):
            try:
                raw = self._batch([panel[i] for i in indexes], kind, horizon, max(lengths[i] for i in indexes))
            except torch.OutOfMemoryError:
                if len(indexes) == 1:
                    raise
            else:
                out[indexes] = raw.transpose(0, 2, 1, 3)
                return
            torch.cuda.empty_cache()
            mid = len(indexes) // 2
            run(indexes[:mid])
            run(indexes[mid:])

        current = []
        for i in sorted(range(len(panel)), key=lengths.__getitem__):
            if current and (len(current) >= 48 or (len(current) + 1) * count * (lengths[i] + horizon) > 2**20):
                run(current)
                current = []
            current.append(i)
        if current:
            run(current)
        return _deciles(out, self.model.quantile_levels)


class CIProtoMember:
    """ci-proto with full single-target cohorts and supplied calendar metadata."""

    def __init__(self, checkpoint, device, frequency):
        from continuousintelligence import load_model

        if str(device).startswith('cuda'):
            _ci_bound_batches()
        self.model = load_model(Path(checkpoint), device=device)
        self.frequency = frequency

    def __call__(self, panel):
        import torch

        horizon = panel[0].future_dates.size
        targets = panel[0].history.shape[0]
        groups = []
        for role in ('past', 'known'):
            names = tuple(getattr(panel[0], role))
            kinds = [_categories([getattr(row, role)[name] for row in panel], numeric_text=False)
                     for name in names]
            categorical = tuple(i for i, (kind, _) in enumerate(kinds) if kind == 'text')
            tail = horizon if role == 'known' else 0
            matrices = []
            for row in panel:
                values = getattr(row, role)
                rows = []
                for name, (kind, vocabulary) in zip(names, kinds, strict=True):
                    column = values[name][-16384 - tail:]
                    rows.append(_codes(column, vocabulary) if kind == 'text' else np.asarray(column, dtype=float))
                if rows:
                    matrices.append(np.stack(rows))
            groups.append((matrices or None, categorical))
        history = [row.history[:, -16384:] for row in panel]
        timestamps = [np.r_[row.history_dates[-16384:], row.future_dates].astype(np.int64) for row in panel]
        if targets == 1:
            width = max(x.shape[-1] for x in history)
            covariates = [torch.from_numpy(_pad(x, width + h)) if x is not None else None
                          for (x, _), h in zip(groups, (0, horizon), strict=True)]
            output = self.model.forecast(torch.from_numpy(_pad([x[0] for x in history], width)),
                                         horizon=horizon, past_covariates=covariates[0], future_covariates=covariates[1],
                                         categorical_past=groups[0][1], categorical_future=groups[1][1],
                                         timestamps=torch.from_numpy(_pad(timestamps, width + horizon, np.iinfo(np.int64).min)),
                                         freq=self.frequency, series_ids=[row.key for row in panel],
                                         static_features={name: [row.static[name] for row in panel]
                                                          for name in panel[0].static} or None)
            out = output.numpy().transpose(0, 2, 1)[:, :, None, :]
        else:
            batch = []
            for i, (prefix, clock) in enumerate(zip(history, timestamps, strict=True)):
                covariates = [torch.from_numpy(x[i]) if x is not None else None for x, _ in groups]
                batch.append(self.model.forecast(torch.from_numpy(prefix.copy()), horizon=horizon,
                                                 past_covariates=covariates[0], future_covariates=covariates[1],
                                                 categorical_past=groups[0][1], categorical_future=groups[1][1],
                                                 timestamps=torch.from_numpy(clock), freq=self.frequency).numpy())
            out = np.asarray(batch).transpose(0, 3, 1, 2)
        return _deciles(out, self.model.quantiles.tolist())

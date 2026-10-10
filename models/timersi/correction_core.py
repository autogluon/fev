from dataclasses import dataclass
import warnings
import numpy as np
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
LEVELS = np.arange(1, 10) / 10
GRID = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
FEATURES = ['level_gap', 'last_gap', 'drift_gap', 'seasonal_gap', 'recent_change', 'forecast_change', 'lead', 'lower_width', 'upper_width', 'past_volatility', 'dow_sin', 'dow_cos', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos']
CANDIDATES = ['baseline', 'shape-blend', 'semantic-rule', 'conditional-ridge', 'conditional-lgbm', 'calibrated-spread']
VARIANTS = ['baseline', 'raw', *CANDIDATES[1:], 'online-select', 'online-top3']

def finite_median(a, axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        value = np.nanmedian(a, axis=axis)
    return np.nan_to_num(value)

def pinball(y, q):
    e = y[..., None] - q
    return 2 * np.abs(e * ((e <= 0) - LEVELS))

@dataclass
class View:
    point: np.ndarray
    quantiles: np.ndarray
    x: np.ndarray
    unit: np.ndarray
    expert: np.ndarray
    last: np.ndarray
    nonnegative: np.ndarray
    monotone: np.ndarray

def make_view(p, q, past, lengths, scale, seasonality, calendar):
    N, H, D = p.shape
    last = np.zeros((N, D))
    level = last.copy()
    change = last.copy()
    vol = last.copy()
    drift = p.copy()
    seasonal = p.copy()
    nonnegative = np.zeros((N, D), bool)
    monotone = nonnegative.copy()
    ends = np.cumsum(lengths)
    for i, end in enumerate(ends):
        hist = past[end - lengths[i]:end]
        tail = hist[-min(len(hist), max(8, min(64, 2 * H))):]
        n = len(tail)
        half = max(1, n // 2)
        level[i] = finite_median(tail)
        last[i] = np.array([v[np.isfinite(v)][-1] if np.isfinite(v).any() else 0 for v in tail.T])
        recent = finite_median(tail[-half:])
        previous = finite_median(tail[:half])
        change[i] = recent - previous
        slope = change[i] / max(1, n - half)
        drift[i] = recent + (np.arange(1, H + 1)[:, None] + (half - 1) / 2) * slope
        vol[i] = np.nan_to_num(np.nanstd(tail, axis=0))
        s = max(1, int(seasonality))
        if s > 1 and len(hist) >= s:
            for lead in range(H):
                positions = len(hist) - s + lead % s - s * np.arange(min(4, len(hist) // s))
                seasonal[i, lead] = finite_median(hist[positions])
        else:
            seasonal[i] = drift[i]
        nonnegative[i] = np.all((tail >= 0) | ~np.isfinite(tail), axis=0)
        diff = np.diff(tail, axis=0)
        monotone[i] = np.mean((diff >= -1e-08) | ~np.isfinite(diff), axis=0) >= 0.98 if n > 1 else False
    unit = np.maximum(np.where(np.isfinite(scale), scale, 0.0), np.maximum(vol * 0.05, 1e-06))
    centre = q[..., 4]

    def broad(v):
        return np.broadcast_to(v[:, None, :], (N, H, D))
    un = unit[:, None, :]
    lead = np.broadcast_to(np.linspace(0, 1, H)[None, :, None], (N, H, D))
    fields = [(broad(level) - centre) / un, (broad(last) - centre) / un, (drift - centre) / un, (seasonal - centre) / un, broad(change) / un, broad(centre[:, -1] - centre[:, 0]) / un, lead, (centre - q[..., 0]) / un, (q[..., -1] - centre) / un, broad(vol) / un]
    for j in range(calendar.shape[-1]):
        fields.append(np.broadcast_to(calendar[:, :, j, None], (N, H, D)))
    x = np.clip(np.nan_to_num(np.stack(fields, axis=-1)), -20, 20)
    expert = seasonal if int(seasonality) > 1 else drift
    return View(p, q, x, unit, expert, last, nonnegative, monotone)

class Reviser:

    def __init__(self, task_name):
        self.task_name = task_name
        self.history = []

    def closed(self, start):
        return [r for r in self.history if np.datetime64(r['end']) < np.datetime64(start)]

    def predict(self, view, start, recipe):
        p, q = (view.point, view.quantiles)
        N, H, D = p.shape
        old = self.closed(start)
        output = {'baseline': (p, q)}
        info = {'closed_windows': [r['window'] for r in old], 'fit': [], 'current_truth_used': False}
        delta = 0.25 * (view.expert - q[..., 4])
        output['shape-blend'] = (p + delta, q + delta[..., None])
        pp, qq = (p.copy(), q.copy())
        positive = self.task_name.startswith(('solar', 'rossmann', 'rohlik', 'favorita', 'uk_covid'))
        if positive:
            allowed = view.nonnegative[:, None, :]
            pp = np.where(allowed, np.maximum(pp, 0), pp)
            qq = np.where(allowed[..., None], np.maximum(qq, 0), qq)
        if 'cumulative' in self.task_name:
            allowed = view.monotone[:, None, :]
            pp = np.where(allowed, np.maximum.accumulate(np.maximum(pp, view.last[:, None, :]), axis=1), pp)
            qq = np.where(allowed[..., None], np.maximum.accumulate(np.maximum(qq, view.last[:, None, :, None]), axis=1), qq)
        output['semantic-rule'] = (pp, qq)
        shifted = {name: np.zeros_like(p) for name in ('conditional-ridge', 'conditional-lgbm')}
        scales = np.ones((D, 2))
        models = {}
        for d in range(D):
            train = [r['train'][d] for r in old[-4:]]
            if not train:
                continue
            X = np.concatenate([r['x'] for r in train])
            y = np.concatenate([r['residual'] for r in train])
            keep = np.isfinite(y) & np.isfinite(X).all(axis=1)
            X, y = (X[keep], y[keep])
            if len(y) < 64:
                continue
            if len(y) > 20000:
                select = np.linspace(0, len(y) - 1, 20000, dtype=int)
                X, y = (X[select], y[select])
            y = np.clip(y, -10, 10)
            xx = view.x[:, :, d].reshape(-1, len(FEATURES))
            ridge = Ridge(alpha=max(1.0, 0.05 * len(y))).fit(X, y)
            gbm = LGBMRegressor(objective='quantile', alpha=0.5, n_estimators=60, num_leaves=7, max_depth=3, learning_rate=0.05, min_child_samples=max(16, min(80, len(y) // 20)), reg_lambda=2.0, verbosity=-1, n_jobs=1, random_state=74).fit(X, y)
            for key, model in [('conditional-ridge', ridge), ('conditional-lgbm', gbm)]:
                pred = model.predict(xx)
                shifted[key][:, :, d] = 0.5 * np.clip(pred.reshape(N, H), -5.0, 5.0) * view.unit[:, d, None]
            models[str(d)] = {'ridge_coef': ridge.coef_.tolist(), 'ridge_intercept': float(ridge.intercept_), 'lgbm': gbm.booster_.model_to_string()}
            info['fit'].append({'target': d, 'rows': len(y), 'windows': [r['window'] for r in old[-4:]], 'latest_training_end': old[-1]['end']})
        for key, delta in shifted.items():
            output[key] = (p + delta, q + delta[..., None])
        scales = np.asarray(recipe['interval_scales'], float)
        sq = q.copy()
        centre = q[..., 4:5]
        sq[..., :4] = centre + scales[None, None, :, 0, None] * (q[..., :4] - centre)
        sq[..., 5:] = centre + scales[None, None, :, 1, None] * (q[..., 5:] - centre)
        output['calibrated-spread'] = (p, sq)
        selected, top3 = (recipe['correction_selected'], recipe['correction_members'])
        output['online-select'] = output[selected]
        output['online-top3'] = tuple((np.mean([output[k][j] for k in top3], axis=0) for j in (0, 1)))
        info.update(selected=selected, top3=top3, lower_upper_scales=scales.tolist(), fitted_targets=len(info['fit']), models=models)
        return (output, info)

    def observe(self, view, truth, window, end, scores):
        train = []
        for d in range(truth.shape[2]):
            un = view.unit[:, d, None]
            x = view.x[:, :, d].reshape(-1, len(FEATURES))
            yy = (truth[:, :, d] / un).reshape(-1)
            qq = (view.quantiles[:, :, d] / un[..., None]).reshape(-1, 9)
            residual = (truth[:, :, d] - view.quantiles[:, :, d, 4]) / un
            ix = np.linspace(0, len(yy) - 1, min(10000, len(yy)), dtype=int)
            train.append({'x': x[ix], 'truth_normalized': yy[ix], 'q_normalized': qq[ix], 'residual': residual.reshape(-1)[ix]})
        self.history.append({'window': window, 'end': end, 'train': train, 'scores': scores})
        self.history = self.history[-8:]

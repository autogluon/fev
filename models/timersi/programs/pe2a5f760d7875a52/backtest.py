import numpy as np
QL = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

def auxiliary_plan(L, H, offsets, min_context):
    plan = []
    for off in offsets:
        if off < H:
            continue
        end = L - off
        if end < min_context or end + H > L:
            continue
        plan.append({'offset': int(off), 'end': int(end), 'context': int(end)})
    return plan

def replay_scores(truth, obs, point, quant):
    truth = np.asarray(truth, float)
    obs = np.asarray(obs, bool)
    point = np.asarray(point, float)
    quant = np.asarray(quant, float)
    good = obs & np.isfinite(truth) & np.isfinite(point) & (truth > 0) & (point > 0)
    out = {'n': int(good.sum())}
    if not good.any():
        return out
    med = quant[:, :, 4]
    safe = good & (med > 0)
    out['log_ratio_point'] = np.where(good, np.log(np.maximum(truth, 1e-09) / np.maximum(point, 1e-09)), np.nan)
    out['log_ratio_med'] = np.where(safe, np.log(np.maximum(truth, 1e-09) / np.maximum(med, 1e-09)), np.nan)
    out['below'] = np.where(good[:, :, None], truth[:, :, None] <= quant, np.nan)
    d = truth[:, :, None] - quant
    out['pinball'] = np.where(good[:, :, None], np.maximum(QL * d, (QL - 1) * d), np.nan)
    out['good'] = good
    return out

def nan_mean(a, axis=None):
    a = np.asarray(a, float)
    if a.size == 0 or not np.isfinite(a).any():
        return np.nan if axis is None else np.full(np.delete(a.shape, axis), np.nan)
    with np.errstate(invalid='ignore'):
        return np.nanmean(a, axis=axis)

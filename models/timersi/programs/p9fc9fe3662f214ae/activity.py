import numpy as np
EPS = 0.001

def _logit(p):
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def activity_mask(history, floor_values):
    tol = 1e-09 * np.maximum(1.0, np.abs(floor_values))
    return history > floor_values[:, None] + tol[:, None]

def detect_period(act, pmax=30, window=4096, min_acf=0.1, default=None):
    n, L = act.shape
    w = min(window, L)
    a = act[:, L - w:].astype(np.float64)
    a = a - a.mean(axis=1, keepdims=True)
    var = (a * a).mean(axis=1) + 1e-12
    best_p = np.full(n, 0, dtype=int)
    best_s = np.full(n, -np.inf)
    for p in range(2, min(pmax, w // 4) + 1):
        s = (a[:, :-p] * a[:, p:]).mean(axis=1) / var
        upd = s > best_s
        best_s = np.where(upd, s, best_s)
        best_p = np.where(upd, p, best_p)
    ok = best_s >= min_acf
    if default is not None:
        best_p = np.where(ok, best_p, default)
    return (best_p, best_s, ok)

def phase_probability(act, periods, horizon, windows=(300, 1200), half_life=300.0):
    n, L = act.shape
    acc = np.zeros((n, horizon))
    used = 0
    for win in windows:
        w = min(win, L)
        if w < 4 * horizon // 3:
            continue
        seg = act[:, L - w:].astype(np.float64)
        wt = 0.5 ** (np.arange(w)[::-1] / half_life)
        pos = np.arange(L - w, L)
        for i in range(n):
            p = int(periods[i])
            if p < 2:
                acc[i, :] += _logit(np.full(horizon, (seg[i] * wt).sum() / wt.sum()))
                continue
            idx = pos % p
            for ph in range(p):
                m = idx == ph
                if not m.any():
                    continue
                val = (seg[i, m] * wt[m]).sum() / wt[m].sum()
                hh = np.arange(horizon)
                sel = (L + hh) % p == ph
                acc[i, sel] += _logit(val)
        used += 1
    if used == 0:
        return None
    return _sigmoid(acc / used)

def quantile_function(cl, floor_values, levels, grid):
    n, h, _ = cl.shape
    vals = np.concatenate([floor_values[:, None, None] * np.ones((n, h, 1)), cl, cl[:, :, -1:]], axis=2)
    u = np.clip(levels, 0.0, 1.0)
    j = np.clip(np.searchsorted(grid, u.ravel(), side='right') - 1, 0, len(grid) - 2)
    j = j.reshape(n, h)
    g0, g1 = (grid[j], grid[j + 1])
    w = np.where(g1 > g0, (u - g0) / np.maximum(g1 - g0, 1e-12), 0.0)
    v0 = np.take_along_axis(vals, j[:, :, None], 2)[:, :, 0]
    v1 = np.take_along_axis(vals, (j + 1)[:, :, None], 2)[:, :, 0]
    return v0 * (1 - w) + v1 * w

def recompose(cl, floor_values, p_new, p_native, q_levels):
    n, h, _ = cl.shape
    grid = np.concatenate([[0.0], q_levels, [1.0]])
    out = np.empty((n, h, len(q_levels)))
    p_t = np.clip(p_native, EPS, 1.0)
    p_n = np.clip(p_new, 0.0, 1.0)
    for j, t in enumerate(q_levels):
        u = np.where(t <= 1.0 - p_n, 0.0, 1.0 - p_t + (t - (1.0 - p_n)) / np.maximum(p_n, 1e-09) * p_t)
        out[:, :, j] = quantile_function(cl, floor_values, u, grid)
    out = np.maximum(out, floor_values[:, None, None])
    return np.sort(out, axis=-1)

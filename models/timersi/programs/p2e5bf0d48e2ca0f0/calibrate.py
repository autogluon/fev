import numpy as np

def moving_level(path, window=9):
    path = np.asarray(path, float)
    half = window // 2
    pad = np.pad(path, ((0, 0), (half, half)), mode='edge')
    kernel = np.ones(window) / window
    return np.vstack([np.convolve(pad[d], kernel, mode='valid') for d in range(path.shape[0])])

def drift_shrink_shift(point, anchor, gamma, ramp_start=1, ramp_end=14, window=9):
    H = point.shape[1]
    h = np.arange(1, H + 1, dtype=float)
    ramp = np.clip((h - ramp_start) / max(ramp_end - ramp_start, 1), 0.0, 1.0)[None, :]
    level = moving_level(point, window)
    return -gamma * ramp * (level - anchor[:, None])

def envelope_guard(point, quantiles, hist, scale, base=1.5, growth=1.0, strength=0.6):
    H = point.shape[1]
    h = np.arange(1, H + 1, dtype=float)
    allow = (base + growth * np.sqrt(h / 7.0))[None, :] * scale[:, None]
    upper = hist.max(axis=1)[:, None] + allow
    lower = hist.min(axis=1)[:, None] - allow

    def squash(x, u, l):
        return x - strength * np.maximum(x - u, 0.0) + strength * np.maximum(l - x, 0.0)
    point = squash(point, upper, lower)
    quantiles = squash(quantiles, upper[..., None], lower[..., None])
    return (point, quantiles)

def tail_widen(quantiles, amount=0.12, start=14):
    H = quantiles.shape[1]
    h = np.arange(1, H + 1, dtype=float)
    w = 1.0 + amount * np.clip((h - start) / max(H - start, 1), 0.0, 1.0)
    med = quantiles[..., 4:5]
    return med + (quantiles - med) * w[None, :, None]

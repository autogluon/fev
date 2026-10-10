import numpy as np
LOG_FLOOR_RATIO = 0.001

class Transform:

    def __init__(self, kind, shift=0.0):
        self.kind = kind
        self.shift = float(shift)

    def forward(self, y):
        y = np.asarray(y, dtype=float)
        if self.kind == 'log':
            return np.log(np.maximum(y + self.shift, 1e-12))
        return y

    def inverse(self, z):
        z = np.asarray(z, dtype=float)
        if self.kind == 'log':
            return np.exp(np.clip(z, -60.0, 60.0)) - self.shift
        return z

def choose_transform(history, observed=None):
    h = np.asarray(history, dtype=float).ravel()
    if observed is not None:
        m = np.asarray(observed).ravel().astype(bool)
        if m.shape == h.shape:
            h = h[m]
    h = h[np.isfinite(h)]
    if h.size == 0:
        return Transform('identity')
    positive = float(h.min()) > 0.0
    if positive:
        span = float(h.max())
        if float(h.min()) > LOG_FLOOR_RATIO * max(span, 1e-12):
            return Transform('log')
    return Transform('identity')

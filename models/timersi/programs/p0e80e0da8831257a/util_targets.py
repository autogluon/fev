import numpy as np

def degenerate_mask(hist, obs, tol=0.0):
    hist = np.asarray(hist, float)
    obs = np.asarray(obs, bool)
    out = []
    for d in range(hist.shape[0]):
        m = obs[d] & np.isfinite(hist[d])
        if m.sum() == 0:
            out.append(True)
            continue
        vals = hist[d][m]
        out.append(bool(np.nanmax(vals) - np.nanmin(vals) <= tol))
    return out

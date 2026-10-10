import numpy as np
HOL = {(1, 1), (1, 6), (4, 25), (5, 1), (6, 2), (8, 15), (11, 1), (12, 8), (12, 25), (12, 26)}
EASTER = {(2004, 4, 11), (2004, 4, 12), (2005, 3, 27), (2005, 3, 28)}
Z9 = None

def _z9():
    global Z9
    if Z9 is None:
        from scipy.stats import norm
        Z9 = norm.ppf(np.arange(0.1, 0.91, 0.1))
    return Z9

def _vac(m, d):
    if m == 8:
        return 1.0 if 7 <= d <= 22 else 0.6
    return 0.0

def features(timestamps, kf_rows, kf_names):
    import pandas as pd
    ts = pd.to_datetime(pd.Series(np.asarray(timestamps)))
    h = ts.dt.hour.values.astype(float)
    dow = ts.dt.dayofweek.values.astype(float)
    cols = [np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24), np.sin(4 * np.pi * h / 24), np.cos(4 * np.pi * h / 24), (dow >= 5).astype(float), (dow == 6).astype(float), np.array([1.0 if (m, d) in HOL or (y, m, d) in EASTER else 0.0 for y, m, d in zip(ts.dt.year, ts.dt.month, ts.dt.day)]), np.array([_vac(m, d) for m, d in zip(ts.dt.month, ts.dt.day)])]
    if kf_rows is not None and len(kf_rows):
        import pandas as pd
        for row in np.asarray(kf_rows, float):
            x = pd.Series(row).interpolate(limit_direction='both')
            cols.append(x.values)
            cols.append(x.rolling(24, min_periods=1).mean().values)
    X = np.column_stack(cols)
    return np.nan_to_num(X, nan=0.0)

def fit_predict(X, y, obs, cut, pred_lo, pred_hi, anchor_win=672, anchor_w=0.5):
    from sklearn.ensemble import HistGradientBoostingRegressor
    y = np.asarray(y, float)
    m = np.asarray(obs, bool)[:cut] & np.isfinite(y[:cut]) & (y[:cut] > 0)
    n = int(m.sum())
    if n < 400:
        return None
    ly = np.log(y[:cut][m])
    gb = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.06, min_samples_leaf=40, random_state=0)
    gb.fit(X[:cut][m], ly)
    fit_res = ly - gb.predict(X[:cut][m])
    sigma = float(np.clip(np.std(fit_res), 0.15, 1.5))
    idx = np.where(m)[0]
    ridx = idx[idx >= cut - anchor_win]
    anchor = float(np.mean(ly[-len(ridx):] - gb.predict(X[ridx]))) if len(ridx) > 24 else 0.0
    point = np.exp(gb.predict(X[pred_lo:pred_hi]) + anchor_w * anchor)
    return (point, sigma, n)

def pinball(truth, mask, point, sigma):
    q = point[:, None] * np.exp(_z9()[None, :] * sigma)
    t = np.asarray(truth, float)
    m = np.asarray(mask, bool) & np.isfinite(t)
    if m.sum() < 24:
        return np.nan
    d = t[m][:, None] - q[m]
    lv = np.arange(0.1, 0.91, 0.1)
    return float(np.mean(np.maximum(lv * d, (lv - 1) * d)))

def support_fraction(Xpast, Xfut, col_T):
    t_p = Xpast[:, col_T]
    lo, hi = (np.percentile(t_p, 2), np.percentile(t_p, 98))
    t_f = Xfut[:, col_T]
    return float(np.mean((t_f >= lo) & (t_f <= hi)))

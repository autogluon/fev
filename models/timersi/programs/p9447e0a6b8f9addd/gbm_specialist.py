import numpy as np
try:
    import lightgbm as lgb
    HAVE_LGB = True
except Exception:
    HAVE_LGB = False
PARAMS = dict(objective='huber', alpha=0.9, num_leaves=7, max_depth=4, learning_rate=0.08, n_estimators=120, min_child_samples=15, subsample=0.9, subsample_freq=1, colsample_bytree=0.9, reg_lambda=1.0, n_jobs=2, verbose=-1)
CLIP = 0.3
HOLDOUT = 48
MIN_TRAIN_ROWS = 220
GAIN_REQ = 0.02
GAMMAS = (0.5, 1.0)

def _features(dyn, months, days, dom_end, n):
    dow = dyn['DayOfWeek']
    openv = dyn['Open']
    closed = openv < 0.5
    since = np.zeros(n)
    till = np.zeros(n)
    c = 7.0
    for t in range(n):
        c = 0.0 if closed[t] else min(c + 1.0, 7.0)
        since[t] = c
    c = 7.0
    for t in range(n - 1, -1, -1):
        c = 0.0 if closed[t] else min(c + 1.0, 7.0)
        till[t] = c
    sth = np.unique(dyn['StateHoliday'], return_inverse=True)[1].astype(float)
    feats = np.column_stack([dow, dyn['Promo'], dyn['SchoolHoliday'], sth, months, days, days / np.maximum(dom_end, 1.0), since, till, dyn['Promo'] * dow])
    return feats

def fit_specialist(dyn, months, days, dom_end, y_hist, L, H, ridge_fit, train_mask):
    if not HAVE_LGB:
        return None
    n = len(months)
    X = _features(dyn, months, days, dom_end, n)
    tr = np.asarray(train_mask, bool).copy()
    tr[L:] = False
    rows = np.nonzero(tr)[0]
    if len(rows) < MIN_TRAIN_ROWS + 30:
        return None
    resid = np.full(n, np.nan)
    resid[rows] = np.log(np.maximum(y_hist[rows], 1.0) / np.maximum(ridge_fit[rows], 1.0))
    cut = L - HOLDOUT
    tr_rows = rows[rows < cut]
    ho_rows = rows[rows >= cut]
    if len(tr_rows) < MIN_TRAIN_ROWS or len(ho_rows) < 12:
        return None
    cat = [0, 3, 4]
    mdl = lgb.LGBMRegressor(**PARAMS)
    mdl.fit(X[tr_rows], resid[tr_rows], categorical_feature=cat)
    corr_ho = np.clip(mdl.predict(X[ho_rows]), -CLIP, CLIP)
    base_err = np.abs(resid[ho_rows])
    gains = {}
    best_gamma, best_gain = (0.0, 0.0)
    for g in GAMMAS:
        err = np.abs(resid[ho_rows] - g * corr_ho)
        gain = 1.0 - err.mean() / max(base_err.mean(), 1e-09)
        gains['gamma=%.1f' % g] = round(float(gain), 4)
        if gain > best_gain:
            best_gain, best_gamma = (gain, g)
    if best_gain < GAIN_REQ:
        return {'gamma': 0.0, 'corr_future': np.zeros(H), 'holdout_gains': gains, 'train_rows': int(len(tr_rows)), 'holdout_rows': int(len(ho_rows))}
    mdl = lgb.LGBMRegressor(**PARAMS)
    mdl.fit(X[rows], resid[rows], categorical_feature=cat)
    corr = np.clip(mdl.predict(X[L:L + H]), -CLIP, CLIP)
    return {'gamma': float(best_gamma), 'corr_future': corr, 'holdout_gains': gains, 'train_rows': int(len(rows)), 'holdout_rows': int(len(ho_rows))}

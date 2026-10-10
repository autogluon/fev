import numpy as np
LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
RISK_TABLE = ((1, 0.125), (2, 0.11), (5, 0.275), (13, 0.864), (10 ** 9, 0.696))
HOLIDAY_RUN_RISK = 0.044

def trailing_zero_risk(history, holiday=None):
    y = np.asarray(history, float)
    n = len(y)
    run = 0
    while run < n and y[n - 1 - run] <= 0:
        run += 1
    if run == 0 or run == n:
        return 0.0
    codes = np.asarray(holiday, float)[n - run:n] if holiday is not None else np.zeros(run)
    if np.all(codes != 0):
        return HOLIDAY_RUN_RISK
    for limit, value in RISK_TABLE:
        if run <= limit:
            return value
    return 0.0

def apply_mixture(point, quantiles, risk):
    if risk <= 0:
        return (point, quantiles)
    h = len(point)
    new_q = np.zeros_like(quantiles)
    for t in range(h):
        for j, tau in enumerate(LEVELS):
            if tau <= risk:
                new_q[t, j] = 0.0
            else:
                new_q[t, j] = np.interp((tau - risk) / (1.0 - risk), LEVELS, quantiles[t])
    old_mid = quantiles[:, 4]
    factor = np.where(old_mid > 0, new_q[:, 4] / np.maximum(old_mid, 1e-09), 0.0)
    return (point * factor, new_q)

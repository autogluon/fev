import datetime as _dt
import numpy as np
from calendar_ops import parse_day
COLD_SCALE = 0.75
LAM_MASS = 3.0
UP_WIDEN = 1.3
LO_WIDEN_FRAC = 0.25
PRIOR_REF = 0.15

def _day_factor(d):
    m, day = (d.month, d.day)
    if m == 12:
        if 6 <= day <= 12:
            return 0.045
        if 13 <= day <= 16:
            return 0.14
        if 17 <= day <= 23:
            return 0.3
        if day == 24:
            return 0.35
        if day == 25:
            return 0.0
        if 26 <= day <= 30:
            return 0.2
        if day == 31:
            return 0.15
    if m == 1:
        if day == 1:
            return 0.0
        if 2 <= day <= 5:
            return 0.08
    return 0.0

def fixed_prior_log(ts_future):
    out = []
    for ts in ts_future:
        end = parse_day(ts)
        tot = sum((_day_factor(end - _dt.timedelta(days=back)) for back in range(7)))
        out.append(COLD_SCALE * tot / 7.0)
    return np.asarray(out, float)

def evidence_lambda(fixed_mass):
    return float(np.clip(fixed_mass / LAM_MASS, 0.0, 1.0))

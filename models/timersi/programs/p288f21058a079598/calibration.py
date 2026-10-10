import numpy as np
BASE_SHRINK = 0.7
SPEED_REF = 0.03
ROUGH_REF = 0.15
SHORT_REF = 3.0
TAIL_SHRINK = 0.85

def tail_profile(tail=TAIL_SHRINK):
    prof = np.ones(9)
    prof[0] = prof[8] = tail
    prof[1] = prof[7] = 1.0 + (tail - 1.0) * 0.5
    return prof

def operating_state(history, horizon):
    y = np.asarray(history, dtype=float)
    y = y[np.isfinite(y)]
    if y.size < 4:
        return {'speed': 1.0, 'rough': 1.0, 'short': 1.0}
    ly = np.log(np.maximum(y, 1e-06))
    n = ly.size
    w = int(min(2 * horizon, n - 1))
    speed = abs((ly[-1] - ly[-1 - w]) / max(w, 1))
    d = np.diff(ly)
    tail = d[-int(min(d.size, 4 * horizon)):]
    rough = (np.percentile(tail, 75) - np.percentile(tail, 25)) / 1.349 if tail.size >= 4 else 1.0
    short = max(0.0, (SHORT_REF * horizon - n) / (SHORT_REF * horizon))
    return {'speed': float(speed), 'rough': float(rough), 'short': float(short)}

def shrink_factor(history, horizon, base=BASE_SHRINK, calendar_risk=0.0):
    st = operating_state(history, horizon)
    u = max(min(1.0, st['speed'] / SPEED_REF), min(1.0, st['rough'] / ROUGH_REF), min(1.0, st['short']), min(1.0, max(0.0, float(calendar_risk))))
    st = dict(st)
    st['calendar_risk'] = float(calendar_risk)
    return (float(base + (1.0 - base) * u), st)

def recalibrate(point, quantiles, history, horizon, base=BASE_SHRINK, tail=TAIL_SHRINK, calendar_risk=0.0):
    q = np.asarray(quantiles, dtype=float).copy()
    med = q[..., 4:5]
    s, st = shrink_factor(history, horizon, base, calendar_risk)
    openness = (s - base) / max(1e-09, 1.0 - base)
    prof = 1.0 + (tail_profile(tail) - 1.0) * (1.0 - openness)
    q = med + (q - med) * (s * prof)
    q = np.sort(q, axis=-1)
    st = dict(st)
    st['openness'] = float(openness)
    return (np.asarray(point, dtype=float), q, s, st)

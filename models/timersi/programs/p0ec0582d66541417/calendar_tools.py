import numpy as np

def to_datetime64(stamps):
    return np.array([np.datetime64(str(s)[:19], 's') for s in stamps])

def year_fraction(stamps):
    dt = to_datetime64(stamps)
    years = dt.astype('datetime64[Y]')
    start = years.astype('datetime64[s]')
    nxt = (years + 1).astype('datetime64[s]')
    span = (nxt - start).astype('float64')
    span = np.where(span <= 0, 365.0 * 86400.0, span)
    return (dt - start).astype('float64') / span % 1.0

def steps_per_year(stamps):
    dt = to_datetime64(stamps)
    if len(dt) < 3:
        return None
    step = np.median(np.diff(dt).astype('float64'))
    if step <= 0:
        return None
    return 365.2425 * 86400.0 / step

def fourier_design(phase, harmonics):
    cols = [np.ones(len(phase))]
    for h in range(1, harmonics + 1):
        cols.append(np.sin(2.0 * np.pi * h * phase))
        cols.append(np.cos(2.0 * np.pi * h * phase))
    return np.column_stack(cols)

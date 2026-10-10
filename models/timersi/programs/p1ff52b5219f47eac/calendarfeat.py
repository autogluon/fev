import numpy as np

def week_phase(timestamps):
    ts = np.array(timestamps, dtype='datetime64[D]')
    years = ts.astype('datetime64[Y]')
    day_of_year = (ts - years).astype(int)
    nxt = (years + 1).astype('datetime64[D]')
    cur = years.astype('datetime64[D]')
    ylen = (nxt - cur).astype(int).astype(float)
    return day_of_year / ylen

def fourier(timestamps, harmonics=(1, 2)):
    ph = week_phase(timestamps)
    out, names = ([], [])
    for k in harmonics:
        out.append(np.sin(2 * np.pi * k * ph))
        names.append('cal_sin%d' % k)
        out.append(np.cos(2 * np.pi * k * ph))
        names.append('cal_cos%d' % k)
    return (np.array(out), names)

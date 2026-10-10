import numpy as np

def decode(timestamps):
    t = np.asarray(timestamps).astype('datetime64[h]')
    hour = (t - t.astype('datetime64[D]')).astype('timedelta64[h]').astype(int)
    days = t.astype('datetime64[D]').astype(int)
    dow = (days + 3) % 7
    return (hour, dow, days)

import datetime as _dt
import numpy as np
DEFAULT_LAT = 32.8
SOLAR_CONST = 1367.0

def _decl(doy):
    return np.radians(23.45) * np.sin(2.0 * np.pi * (284.0 + doy) / 365.0)

def daily_extraterrestrial(doy, lat_deg=DEFAULT_LAT):
    doy = np.asarray(doy, dtype=float)
    lat = np.radians(float(lat_deg))
    d = _decl(doy)
    ws = np.arccos(np.clip(-np.tan(lat) * np.tan(d), -1.0, 1.0))
    e0 = 1.0 + 0.033 * np.cos(2.0 * np.pi * doy / 365.0)
    h0 = 24.0 * 3600.0 / np.pi * SOLAR_CONST * e0 * (np.cos(lat) * np.cos(d) * np.sin(ws) + ws * np.sin(lat) * np.sin(d))
    return np.maximum(h0, 1.0) / 1000000.0

def day_length(doy, lat_deg=DEFAULT_LAT):
    doy = np.asarray(doy, dtype=float)
    lat = np.radians(float(lat_deg))
    d = _decl(doy)
    return 24.0 / np.pi * np.arccos(np.clip(-np.tan(lat) * np.tan(d), -1.0, 1.0))

def clim_air_temp(doy, mean_c=17.5, amp_c=9.5, peak_doy=200.0):
    doy = np.asarray(doy, dtype=float)
    return mean_c - amp_c * np.cos(2.0 * np.pi * (doy - peak_doy + 182.5) / 365.0 + np.pi)

def _parse(ts):
    s = str(ts)[:10]
    return _dt.date(int(s[0:4]), int(s[5:7]), int(s[8:10]))

def week_days(timestamps, first_bin_partial=True):
    dates = [_parse(t) for t in timestamps]
    step = 7
    if len(dates) > 1:
        step = max(1, min(31, (dates[1] - dates[0]).days))
    out = []
    for j, end in enumerate(dates):
        days = [end - _dt.timedelta(days=k) for k in range(step)]
        if j == 0 and first_bin_partial:
            start_year = _dt.date(end.year, 1, 1)
            if days[-1] < start_year <= end:
                days = [d for d in days if d >= start_year]
        out.append(np.array([d.timetuple().tm_yday for d in days], dtype=float))
    return out

def h0_week_sum(timestamps, lat_deg=DEFAULT_LAT):
    return np.asarray([np.sum(daily_extraterrestrial(d, lat_deg)) for d in week_days(timestamps)], dtype=float)

def insolation_index(timestamps, lat_deg=DEFAULT_LAT, alpha=0.8):
    vals = []
    for doys in week_days(timestamps):
        vals.append(np.sum(daily_extraterrestrial(doys, lat_deg) ** float(alpha)))
    return np.asarray(vals, dtype=float)

def daylength_index(timestamps, lat_deg=DEFAULT_LAT):
    return np.asarray([np.sum(day_length(d, lat_deg)) for d in week_days(timestamps)], float)

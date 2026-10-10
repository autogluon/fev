import numpy as np
try:
    import pandas as pd
except Exception:
    pd = None
MIN_CONTEXT = 420
ZERO_SPARSE_FRACTION = 0.5
SPIKE_RATIO = 3.0
GUARD_WEIGHT = 0.5

def _as_dates(stamps):
    return pd.to_datetime(np.asarray(stamps).astype('datetime64[ns]'))

def _leading_zero_trim(y, observed=None):
    y = np.asarray(y, float)
    nz = np.nonzero(np.abs(y) > 0)[0]
    if len(nz) == 0:
        return 0
    first = int(nz[0])
    if first < 30:
        return 0
    tail = y[first:first + 60]
    if np.count_nonzero(tail) < 5:
        for j in nz:
            seg = y[j:j + 60]
            if np.count_nonzero(seg) >= 5:
                first = int(j)
                break
    return max(0, first - 7)

def _seasonal_analog(y, dates, fdates, halfwin=10):
    if pd is None:
        return None
    s = pd.Series(np.asarray(y, float), index=pd.DatetimeIndex(dates))
    acc = np.zeros(len(fdates))
    wsum = np.zeros(len(fdates))
    for lag, w in ((364, 1.0), (728, 0.5)):
        if len(s) <= lag + 91:
            continue
        cur = s.values[-91:].mean()
        prev = s.values[-91 - lag:-lag].mean()
        ratio = float(np.clip((cur + 0.5) / (prev + 0.5), 1.0 / 3.0, 3.0))
        for t, d in enumerate(fdates):
            target = d - pd.Timedelta(days=lag)
            sel = s[(s.index >= target - pd.Timedelta(days=halfwin)) & (s.index <= target + pd.Timedelta(days=halfwin))]
            if len(sel):
                acc[t] += w * float(np.median(sel.values)) * ratio
                wsum[t] += w
    if not np.any(wsum > 0):
        return None
    return np.where(wsum > 0, acc / np.maximum(wsum, 1e-09), np.nan)

def preprocess(view, card):
    y = np.asarray(view['target_history'], float)
    state = {'N': y.shape[0], 'H': int(view['horizon']), 'L': int(view['cutoff_index']), 'hist': y, 'stamps': view['timestamps']}
    return (view, state)

def engineer(prepared, card, state):
    y = state['hist']
    L = state['L']
    starts = [_leading_zero_trim(y[d]) for d in range(y.shape[0])]
    start = int(min(starts)) if starts else 0
    if L - start < MIN_CONTEXT:
        start = max(0, L - MIN_CONTEXT)
    state['start'] = start
    return (prepared, state)

def select_context(variables, card, state):
    L = state['L']
    H = state['H']
    start = state.get('start', 0)
    C = min(L - start, int(card['limits']['max_context']))
    po = np.asarray(variables['past_features'], float)
    pf = np.asarray(variables['known_features'], float)
    case = {'target_indices': list(range(state['N'])), 'targets': np.asarray(variables['target_history'], float)[:, -C:], 'past_only': po[:, -C:] if len(po) else None, 'known_future': pf[:, -(C + H):] if len(pf) else None, 'past_names': variables['past_names'], 'known_names': variables['known_names'], 'provenance': 'context trimmed at the structural start of activity (C=%d of %d)' % (C, L)}
    return ([case], state)

def postprocess(outputs, cases, state):
    N, H = (state['N'], state['H'])
    point = np.empty((N, H))
    quantiles = np.empty((N, H, 9))
    for output, case in zip(outputs, cases):
        point[case['target_indices']] = np.asarray(output['point'], float)
        quantiles[case['target_indices']] = np.asarray(output['quantiles'], float)
    if pd is not None:
        L = state['L']
        try:
            dates = _as_dates(state['stamps'])
            past_dates, future_dates = (list(dates[:L]), list(dates[L:L + H]))
        except Exception:
            past_dates = future_dates = None
        if future_dates:
            for d in range(N):
                y = state['hist'][d]
                if len(y) < 400:
                    continue
                zf = float(np.mean(y[-365:] == 0))
                if zf <= ZERO_SPARSE_FRACTION:
                    continue
                recent = float(np.mean(y[-56:]))
                nat = float(np.mean(point[d]))
                analog = _seasonal_analog(y, past_dates, future_dates)
                if analog is None:
                    continue
                analog = np.nan_to_num(analog, nan=recent)
                am = float(np.mean(analog))
                spike_up = nat > max(SPIKE_RATIO * recent, 0.5)
                spike_dn = am > max(SPIKE_RATIO * max(nat, recent), 0.5)
                if not (spike_up or spike_dn):
                    continue
                blend = np.exp((1.0 - GUARD_WEIGHT) * np.log(np.maximum(point[d], 0.0) + 0.1) + GUARD_WEIGHT * np.log(np.maximum(analog, 0.0) + 0.1)) - 0.1
                blend = np.clip(blend, 0.0, None)
                ratio = np.where(point[d] > 1e-06, (blend + 1e-09) / (point[d] + 1e-09), 1.0)
                point[d] = point[d] * ratio
                quantiles[d] = np.clip(quantiles[d] * ratio[:, None], 0.0, None)
    return {'point': point, 'quantiles': quantiles}

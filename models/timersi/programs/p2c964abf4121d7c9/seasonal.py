import numpy as np
import pandas as pd
from calendar_es import event_features
EVENT_ORDER = ['adj_after', 'adj_before', 'holy_week', 'eve', 'bridge', 'is_holiday']
MINOR_PRIOR = {'bridge': 0.45, 'eve': 0.4, 'adj_before': 0.1, 'adj_after': 0.12, 'holy_week': 0.35}
APPLIED_MINOR = ('bridge', 'eve', 'holy_week')

def classify(ev):
    lab = pd.Series('', index=ev.index)
    for c in EVENT_ORDER:
        lab[ev[c].values] = c
    return lab

def decompose(y, idx, ev, observed=None, iters=3, win=29):
    y = np.asarray(y, float)
    ok = np.isfinite(y) & (y > 0)
    if observed is not None:
        ok &= np.asarray(observed, bool)
    logy = pd.Series(np.where(ok, np.log(np.where(ok, y, 1.0)), np.nan), index=idx)
    lab = classify(ev.loc[idx])
    normal = (lab == '').values & ok
    dow = idx.dayofweek.values
    dowf = pd.Series(0.0, index=range(7))
    level = pd.Series(np.nan, index=idx)
    for _ in range(iters):
        adj = logy - pd.Series(dowf.reindex(dow).values, index=idx)
        level = adj.where(normal).rolling(win, center=True, min_periods=max(5, win // 4)).median()
        level = level.interpolate(limit_direction='both')
        if not np.isfinite(level.values).all():
            level = pd.Series(np.nanmedian(logy.values), index=idx)
            break
        r = logy - level
        dowf = pd.Series(r.where(normal).values, index=dow).groupby(level=0).median().reindex(range(7)).fillna(0.0)
        dowf -= dowf.mean()
    resid = (logy - level - pd.Series(dowf.reindex(dow).values, index=idx)).values
    return dict(level=level, dowf=dowf, resid=resid, lab=lab.values, normal=normal, weekend=dow >= 5, names=ev.loc[idx, 'holiday'].values, ok=ok)

def fit_effects(dec, k_spec=2.0, k_cls=1.5):
    r, lab, we, nm = (dec['resid'], dec['lab'], dec['weekend'], dec['names'])
    fin = np.isfinite(r)
    eff, support = ({}, {})
    gen = {}
    for w in (False, True):
        sel = fin & (lab == 'is_holiday') & (we == w)
        gen[w] = float(np.median(r[sel])) if sel.sum() >= 2 else np.nan
    if not np.isfinite(gen[False]):
        gen[False] = 0.0
    if not np.isfinite(gen[True]):
        gen[True] = 0.3 * gen[False]
    eff['gen', False], eff['gen', True] = (gen[False], gen[True])
    support['holidays'] = int((fin & (lab == 'is_holiday')).sum())
    for w in (False, True):
        names = set(nm[fin & (lab == 'is_holiday') & (we == w)])
        for name in names:
            sel = fin & (lab == 'is_holiday') & (we == w) & (nm == name)
            n = int(sel.sum())
            eff['h', name, w] = (n * float(np.median(r[sel])) + k_spec * gen[w]) / (n + k_spec)
    for c in MINOR_PRIOR:
        for w in (False, True):
            sel = fin & (lab == c) & (we == w)
            prior = MINOR_PRIOR[c] * gen[w] * (1.0 if not w else 0.5)
            n = int(sel.sum())
            eff[c, w] = (n * float(np.median(r[sel])) + k_cls * prior) / (n + k_cls) if n else prior
    return (eff, support)

def future_log_adjustment(fut_idx, ev, eff, scale=1.0, minor=APPLIED_MINOR):
    lab = classify(ev.loc[fut_idx]).values
    nm = ev.loc[fut_idx, 'holiday'].values
    we = fut_idx.dayofweek.values >= 5
    out = np.zeros(len(fut_idx))
    for j in range(len(fut_idx)):
        c, w = (lab[j], bool(we[j]))
        if c == 'is_holiday':
            out[j] = eff.get(('h', nm[j], w), eff['gen', w])
        elif c in minor:
            out[j] = eff[c, w]
    return out * scale

def past_log_adjustment(past_idx, ev, eff, scale=1.0, minor=APPLIED_MINOR):
    return future_log_adjustment(past_idx, ev, eff, scale, minor)

def clean_history(y, dec, clip=(-1.2, 0.5)):
    y = np.asarray(y, float).copy()
    r = dec['resid']
    is_event = (dec['lab'] != '') & dec['ok'] & np.isfinite(r)
    if not is_event.any():
        return (y, 0)
    corr = np.clip(r[is_event], clip[0], clip[1])
    y[is_event] = y[is_event] * np.exp(-corr)
    return (y, int(is_event.sum()))

def specialist(dec, fut_idx, log_event_adj, level_window=21):
    lv = np.asarray(dec['level'].values, float)
    lv = lv[np.isfinite(lv)]
    if lv.size == 0:
        return None
    base = float(np.median(lv[-level_window:]))
    dowf = dec['dowf'].reindex(fut_idx.dayofweek).values
    return base + dowf + log_event_adj

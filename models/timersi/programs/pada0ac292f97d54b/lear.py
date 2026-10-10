import numpy as np
from sklearn.linear_model import LassoLarsIC
import warnings
warnings.filterwarnings('ignore')

def easter(year):
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = (h + l - 7 * m + 114) % 31 + 1
    return np.datetime64('%04d-%02d-%02d' % (year, month, day))
_HOL_CACHE = {}

def german_holidays(year):
    if year in _HOL_CACHE:
        return _HOL_CACHE[year]
    e = easter(year)
    days = {np.datetime64('%d-01-01' % year), np.datetime64('%d-05-01' % year), np.datetime64('%d-10-03' % year), np.datetime64('%d-12-25' % year), np.datetime64('%d-12-26' % year), np.datetime64('%d-12-24' % year), np.datetime64('%d-12-31' % year), e - np.timedelta64(2, 'D'), e + np.timedelta64(1, 'D'), e + np.timedelta64(39, 'D'), e + np.timedelta64(50, 'D')}
    _HOL_CACHE[year] = days
    return days

def holiday_flag(day_dates):
    out = np.zeros(len(day_dates))
    for j, d in enumerate(day_dates):
        y = d.astype('datetime64[Y]').astype(int) + 1970
        if d in german_holidays(y):
            out[j] = 1.0
    return out

def build_day_grid(y, exog, day_dates):
    return (y, exog, day_dates)

def make_features(P, EX, dow, hol, D, lagdays=(1, 2, 3, 7)):
    feats = [P[D - l] for l in lagdays]
    feats.append(np.array([P[D - 1].max(), P[D - 1].min(), P[D - 1].mean(), P[D - 7].max(), P[D - 7].min(), P[D - 7].mean()]))
    for X in EX:
        feats.append(X[D])
        feats.append(X[D - 1])
        feats.append(X[D - 7])
        feats.append(np.array([X[D].mean(), X[D].max(), X[D].min()]))
    oh = np.zeros(7)
    oh[dow[D]] = 1.0
    feats.append(oh)
    feats.append(np.array([hol[D], hol[D - 1]]))
    return np.concatenate(feats)

class LEAR:

    def __init__(self, windows=(56, 84, 1092, 1456), criterion='aic'):
        self.windows = windows
        self.criterion = criterion

    @staticmethod
    def _scale(A):
        med = np.median(A, axis=0)
        mad = np.median(np.abs(A - med), axis=0) * 1.4826
        mad = np.where(mad < 1e-08, 1.0, mad)
        return (med, mad)

    def fit_predict(self, F, P, D, windows=None, return_each=False):
        windows = sorted(windows or self.windows, reverse=True)
        p = F.shape[1]
        preds = []
        nv = None
        for W in windows:
            lo = max(7, D - W)
            n = D - lo
            if n < 30:
                continue
            Xtr = F[lo:D]
            Ytr = P[lo:D]
            xm, xs = self._scale(Xtr)
            Xs = np.arcsinh((Xtr - xm) / xs)
            xt = np.arcsinh((F[D] - xm) / xs)[None, :]
            ym, ys = self._scale(Ytr)
            Ys = np.arcsinh((Ytr - ym) / ys)
            out = np.empty(24)
            rv = np.empty(24)
            for h in range(24):
                kw = {} if n > p + 10 and nv is None else {'noise_variance': float(nv[h]) if nv is not None else float(np.var(Ys[:, h]) * 0.2)}
                m = LassoLarsIC(criterion=self.criterion, max_iter=250, **kw)
                m.fit(Xs, Ys[:, h])
                out[h] = np.sinh(m.predict(xt)[0]) * ys[h] + ym[h]
                r = Ys[:, h] - m.predict(Xs)
                rv[h] = max(np.var(r), 0.0001)
            if nv is None and n > p + 10:
                nv = rv
            preds.append(out)
        if not preds:
            return P[D - 1].copy()
        if return_each:
            return np.array(preds)
        return np.mean(preds, axis=0)

    def static_fit_predict(self, F, P, lo, end, targets, max_iter=250):
        Xtr = F[lo:end]
        Ytr = P[lo:end]
        if not np.isfinite(Xtr).all() or not np.isfinite(Ytr).all():
            return None
        p = Xtr.shape[1]
        n = end - lo
        xm, xs = self._scale(Xtr)
        Xs = np.arcsinh((Xtr - xm) / xs)
        Xq = np.arcsinh((F[targets] - xm) / xs)
        ym, ys = self._scale(Ytr)
        Ys = np.arcsinh((Ytr - ym) / ys)
        out = np.empty((len(targets), 24))
        for h in range(24):
            kw = {} if n > p + 10 else {'noise_variance': float(np.var(Ys[:, h]) * 0.2)}
            m = LassoLarsIC(criterion=self.criterion, max_iter=max_iter, **kw)
            m.fit(Xs, Ys[:, h])
            out[:, h] = np.sinh(m.predict(Xq)) * ys[h] + ym[h]
        return out

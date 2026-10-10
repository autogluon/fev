import numpy as np
ORDERS = 'total_orders'

def orders_series(view):
    names = list(view.get('known_names') or [])
    if ORDERS not in names:
        return None
    arr = np.asarray(view['known_features'][names.index(ORDERS)], dtype=float)
    if not np.all(np.isfinite(arr)):
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr

def split(view):
    arr = orders_series(view)
    if arr is None:
        return (None, None)
    L = int(view['cutoff_index'])
    H = int(view['horizon'])
    if arr.shape[0] < L + H:
        return (None, None)
    return (arr[:L], arr[L:L + H])

def closed_mask(values, floor_ratio=0.0):
    if values is None:
        return None
    return values <= floor_ratio

def open_level(history, open_past, window=13, fallback=None):
    h = np.asarray(history, dtype=float)
    if open_past is None:
        open_past = np.ones(h.shape[-1], dtype=bool)
    seg = h[-window:]
    msk = open_past[-window:]
    use = seg[msk] if msk.sum() >= 2 else seg
    if use.size == 0:
        use = h if h.size else np.array([0.0])
    val = float(np.median(use))
    if not np.isfinite(val):
        val = float(fallback if fallback is not None else 0.0)
    return val

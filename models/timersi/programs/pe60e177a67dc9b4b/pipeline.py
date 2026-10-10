import numpy as np
from views import MAX_CTX, window
from calibrate import recalibrate
from experts import daytype_quantiles
REQUESTED_CTX = MAX_CTX
EXPERT_WEIGHT = 0.12
N_MATCH_WEEK = 4
SLOT_WIN = 2
LOG_SPACE = True
LOG_FLOOR = 0.5

def preprocess(view, card):
    hist = np.asarray(view['target_history'], dtype=np.float64)
    L = int(view['cutoff_index'])
    state = {'N': hist.shape[0], 'H': int(view['horizon']), 'L': L, 'item': view['item_id'], 'hist': hist, 'first_future_ts': str(view['timestamps'][L])}
    return (view, state)

def engineer(prepared, card, state):
    try:
        state['expert'] = daytype_quantiles(state['hist'], state['L'], state['H'], np.datetime64(state['first_future_ts']), N_MATCH_WEEK, SLOT_WIN)
    except Exception:
        state['expert'] = None
    return (prepared, state)

def select_context(variables, card, state):
    ctx, c = window(state['hist'], state['L'], REQUESTED_CTX)
    state['ctx_len'] = c
    sent = np.log(np.maximum(ctx, LOG_FLOOR)) if LOG_SPACE else ctx
    return ([{'target_indices': list(range(state['N'])), 'targets': sent, 'past_only': None, 'known_future': None, 'past_names': [], 'known_names': [], 'provenance': f"single window ending at cutoff, ctx={c}, space={('log' if LOG_SPACE else 'raw')}"}], state)

def postprocess(outputs, cases, state):
    N, H = (state['N'], state['H'])
    point = np.empty((N, H))
    quant = np.empty((N, H, 9))
    expert = state.get('expert')
    for output, case in zip(outputs, cases):
        idx = case['target_indices']
        q = np.asarray(output['quantiles'], dtype=np.float64)
        if LOG_SPACE:
            q = np.exp(np.clip(q, -20.0, 20.0))
        q = recalibrate(q)
        if expert is not None and np.isfinite(expert).all() and (expert.shape == q.shape):
            q = (1.0 - EXPERT_WEIGHT) * q + EXPERT_WEIGHT * expert[idx]
        q = np.sort(q, axis=-1)
        quant[idx] = q
        point[idx] = q[..., 4]
    return {'point': point, 'quantiles': quant}

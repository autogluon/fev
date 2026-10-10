import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import solar
BLEND_SPECIALIST = 0.6
EXTRA_KNOWN_CHANNELS = True

def preprocess(view, card):
    state = {'N': len(view['target_ids']), 'H': view['horizon']}
    try:
        import pandas as pd
        df, future_ts = solar.frame_from_view(view)
        obs = df['target'].notna().values
        doy = df.index.dayofyear.values
        hour = df.index.hour.values
        cs98 = solar.clearsky_table(df['target'].values[obs], doy[obs], hour[obs], q=0.98)
        dp = solar.daylight_prob(df['target'].values[obs], doy[obs], hour[obs])
        fdoy = pd.DatetimeIndex(future_ts).dayofyear.values
        fhour = pd.DatetimeIndex(future_ts).hour.values
        state['f_cs'] = cs98[fdoy, fhour]
        state['f_dp'] = dp[fdoy, fhour]
        state['nonneg'] = bool(np.nanmin(df['target'].values[obs]) >= 0.0)
        state['_df'] = df
        state['_future_ts'] = future_ts
        full = pd.DatetimeIndex(np.asarray(view['timestamps']))
        state['chan_cs'] = cs98[full.dayofyear.values, full.hour.values]
        state['chan_dp'] = dp[full.dayofyear.values, full.hour.values]
    except Exception as exc:
        state['error_pre'] = repr(exc)
    return (view, state)

def engineer(prepared, card, state):
    if '_df' in state:
        try:
            Q, rows, delta = solar.specialist_quantiles(state['_df'], state['_future_ts'])
            state['spec_q'] = Q
            state['spec_delta'] = delta.tolist()
        except Exception as exc:
            state['error_spec'] = repr(exc)
        state.pop('_df', None)
        state.pop('_future_ts', None)
    return (prepared, state)

def select_context(variables, card, state):
    C = min(variables['cutoff_index'], card['limits']['max_context'])
    H = variables['horizon']
    po = variables['past_features']
    pf = np.asarray(variables['known_features'], dtype=float)
    known_names = list(variables['known_names'])
    if 'chan_cs' in state and EXTRA_KNOWN_CHANNELS:
        extra = np.stack([np.asarray(state['chan_cs'], dtype=float), 100.0 * np.asarray(state['chan_dp'], dtype=float)])
        if len(pf) == 0 or extra.shape[1] == pf.shape[1]:
            pf = np.concatenate([pf, extra], axis=0) if len(pf) else extra
            known_names = known_names + ['clearsky_envelope', 'daylight_probability']
    return ([{'target_indices': list(range(state['N'])), 'targets': variables['target_history'][:, -C:], 'past_only': po[:, -C:] if len(po) else None, 'known_future': pf[:, -(C + H):] if len(pf) else None, 'past_names': variables['past_names'], 'known_names': known_names, 'provenance': 'Reference inputs plus constructed clear-sky envelope / daylight probability known channels, max_context=15360'}], state)

def postprocess(outputs, cases, state):
    N, H = (state['N'], state['H'])
    point = np.empty((N, H))
    quantiles = np.empty((N, H, 9))
    for output, case in zip(outputs, cases):
        point[case['target_indices']] = output['point']
        quantiles[case['target_indices']] = output['quantiles']
    if 'f_cs' in state:
        cs = np.asarray(state['f_cs'], dtype=float)[None, :]
        dp = np.asarray(state['f_dp'], dtype=float)[None, :]
        cap = 1.15 * cs + 50.0
        dark = dp <= 0.0

        def constrain(q, p):
            q = np.minimum(q, cap[:, :, None])
            p = np.minimum(p, cap)
            q = np.where(dark[:, :, None], 0.0, q)
            p = np.where(dark, 0.0, p)
            if state.get('nonneg', False):
                q = np.clip(q, 0.0, None)
                p = np.clip(p, 0.0, None)
            return (np.sort(q, axis=-1), p)
        quantiles, point = constrain(quantiles, point)
        spec = state.get('spec_q')
        if spec is not None and N == 1:
            sq = np.asarray(spec, dtype=float)[None, :, :]
            sq, _ = constrain(sq, np.zeros((1, H)))
            w = BLEND_SPECIALIST
            quantiles = np.sort((1.0 - w) * quantiles + w * sq, axis=-1)
            point = (1.0 - w) * point + w * sq[:, :, 4]
            quantiles, point = constrain(quantiles, point)
    return {'point': point, 'quantiles': quantiles}

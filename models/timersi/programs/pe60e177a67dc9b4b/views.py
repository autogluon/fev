import numpy as np
MAX_CTX = 15360
DAY = 288
WEEK = 7 * DAY

def context_length(L, requested=MAX_CTX):
    return int(max(1, min(requested, MAX_CTX, L)))

def window(series, L, requested=MAX_CTX):
    c = context_length(L, requested)
    return (series[..., L - c:L], c)

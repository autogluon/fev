WINDOW = 288
MIN_WINDOW = 96

def choose_context(available, max_context, window=WINDOW):
    return max(1, min(int(available), int(max_context), int(window)))

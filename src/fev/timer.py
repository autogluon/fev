import inspect
import sys
import time
from collections import defaultdict
from functools import wraps


def line_timing(func):
    src_lines, start_line = inspect.getsourcelines(func)
    filename = inspect.getsourcefile(func) or func.__code__.co_filename
    src_map = {start_line + i: line.rstrip("\n") for i, line in enumerate(src_lines)}

    @wraps(func)
    def wrapper(*args, **kwargs):
        timings_ns = defaultdict(int)
        last_line = None
        last_t = None

        def tracer(frame, event, arg):
            nonlocal last_line, last_t
            if event == "line":
                if frame.f_code is func.__code__:
                    now = time.perf_counter_ns()
                    if last_line is not None and last_t is not None:
                        timings_ns[last_line] += now - last_t
                    last_line = frame.f_lineno
                    last_t = now
            elif event == "return":
                if frame.f_code is func.__code__:
                    now = time.perf_counter_ns()
                    if last_line is not None and last_t is not None:
                        timings_ns[last_line] += now - last_t
            return tracer

        oldtrace = sys.gettrace()
        sys.settrace(tracer)
        try:
            return func(*args, **kwargs)
        finally:
            sys.settrace(oldtrace)

            items = sorted(timings_ns.items(), key=lambda x: x[0])
            total = sum(ns for _, ns in items) or 1

            print(f"\nLine timing for {func.__qualname__} ({filename}:{start_line})")
            print("  % time |   ms   | line | code")
            for lineno, ns in items:
                ms = ns / 1e6
                pct = 100.0 * ns / total
                code = src_map.get(lineno, "").strip()
                print(f"{pct:7.2f} | {ms:7.1f} | {lineno:4d} | {code}")
            print()

    return wrapper

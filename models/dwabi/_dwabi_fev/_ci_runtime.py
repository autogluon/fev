"""Scope the qualified CI-Proto rotary kernel to Forephase GPU calls."""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from threading import Lock

_FIXED = ContextVar("forephase_ci_fixed_rotary", default=False)
_INSTALL = Lock()
_ANGLE_DT_SHA256 = "94e9a49ec8e184df8c2bd9029bba547004d8d124a52de2eed29a213a00848937"


class _RotaryLaunch:
    def __init__(self, original, options):
        self.original = original
        self.options = options

    def __getitem__(self, grid):
        if _FIXED.get():
            return partial(self.original.fn[grid], **self.options)
        return self.original[grid]


def prepare_fixed_rotary(device, backend):
    """Install one dispatcher when the qualified vendor kernel is usable."""
    if backend != "mamba3":
        return False
    import torch
    from continuousintelligence._vendor.mamba3 import angle_dt
    from triton.runtime.autotuner import Autotuner

    if (torch.cuda.get_device_capability(device) != (8, 9)
            or version("continuousintelligence") != "0.1.0"
            or version("triton") != "3.6.0"):
        return False
    try:
        digest = sha256(Path(angle_dt.__file__).read_bytes()).hexdigest()
    except OSError:
        return False
    if digest != _ANGLE_DT_SHA256:
        return False

    with _INSTALL:
        kernel = angle_dt.angle_dt_fwd_kernel
        if isinstance(kernel, _RotaryLaunch):
            return True
        if not isinstance(kernel, Autotuner):
            return False
        entries = [x for x in kernel.configs if x.num_warps == 2 and x.num_stages == 1]
        if len(entries) != 1 or entries[0].pre_hook is not None:
            return False
        angle_dt.angle_dt_fwd_kernel = _RotaryLaunch(kernel, entries[0].all_kwargs())
        return True


@contextmanager
def rotary_scope(enabled):
    marker = _FIXED.set(enabled)
    try:
        yield
    finally:
        _FIXED.reset(marker)
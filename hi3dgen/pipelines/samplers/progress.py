"""Per-step hook of the samplers: live progress reporting and cooperative cancellation.

The caller (the app) installs a callback for the current thread; the samplers call `notify_step` once per sampling step.
The callback may raise `SamplingCancelled` to stop the run between two steps. Without a callback nothing happens, so the
samplers behave exactly as before.
"""
import threading
from contextlib import contextmanager
from typing import Callable, Optional


class SamplingCancelled(Exception):
    """Raised by a step callback to stop sampling between two steps."""


_state = threading.local()


@contextmanager
def step_callback(callback: Optional[Callable[[int, int, str], None]]):
    """Install `callback(step, total, stage)` for the current thread while the block runs."""
    previous = getattr(_state, "callback", None)
    _state.callback = callback
    try:
        yield
    finally:
        _state.callback = previous


@contextmanager
def sampling_stage(name: str):
    """Name the sampling stage that is about to run (shown in the progress text)."""
    previous = getattr(_state, "stage", "")
    _state.stage = name
    try:
        yield
    finally:
        _state.stage = previous


def notify_step(step: int, total: int) -> None:
    callback = getattr(_state, "callback", None)
    if callback is not None:
        callback(step, total, getattr(_state, "stage", ""))

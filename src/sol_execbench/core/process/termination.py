# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Scoped termination handling for owners of resources requiring teardown."""

from __future__ import annotations

import signal
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import FrameType

type _SignalHandler = (
    Callable[[int, FrameType | None], object] | int | signal.Handlers | None
)

TERMINATION_SIGNALS = (signal.SIGTERM, signal.SIGINT)


@dataclass(slots=True, kw_only=True)
class _TerminationState:
    active_depth: int = 0
    unwinding: bool = False
    root_handlers: dict[signal.Signals, _SignalHandler] = field(
        default_factory=dict
    )


_STATE = _TerminationState()


def _ignore_signal(_signum: int, _frame: FrameType | None) -> None:
    """Ignore follow-up signals in Python while letting child execve reset to SIG_DFL."""
    return


@contextmanager
def defer_termination() -> Iterator[None]:
    """Shield bounded teardown from a second interrupt on the main thread."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous: dict[signal.Signals, _SignalHandler] = {
        sig: signal.getsignal(sig) for sig in TERMINATION_SIGNALS
    }
    try:
        for sig in previous:
            signal.signal(sig, _ignore_signal)
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


@contextmanager
def cleanup_on_termination() -> Iterator[None]:
    """Convert SIGTERM/SIGINT to stack unwinding, preserving nesting safety."""
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError(
            "clock_main_thread_required: signal cleanup unavailable"
        )

    def terminate(number: int, _frame: FrameType | None) -> None:
        _STATE.unwinding = True
        for sig in TERMINATION_SIGNALS:
            signal.signal(sig, _ignore_signal)
        raise SystemExit(128 + number)

    if _STATE.active_depth == 0:
        _STATE.root_handlers.clear()
        for sig in TERMINATION_SIGNALS:
            _STATE.root_handlers[sig] = signal.getsignal(sig)
        _STATE.unwinding = False
        for sig in TERMINATION_SIGNALS:
            signal.signal(sig, terminate)
    _STATE.active_depth += 1
    try:
        yield
    finally:
        _STATE.active_depth -= 1
        if _STATE.active_depth == 0:
            try:
                for sig, handler in _STATE.root_handlers.items():
                    signal.signal(sig, handler)
            finally:
                _STATE.root_handlers.clear()
                _STATE.unwinding = False

# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Scoped termination handling for owners of resources requiring teardown."""

from __future__ import annotations

import signal
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from types import FrameType

TERMINATION_SIGNALS = (signal.SIGTERM, signal.SIGINT)


@contextmanager
def defer_termination() -> Iterator[None]:
    """Shield bounded teardown from a second interrupt on the main thread."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = {sig: signal.getsignal(sig) for sig in TERMINATION_SIGNALS}
    try:
        for sig in previous:
            signal.signal(sig, signal.SIG_IGN)
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


@contextmanager
def cleanup_on_termination() -> Iterator[None]:
    """Convert SIGTERM/SIGINT to stack unwinding, preserving previous handlers."""
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError(
            "clock_main_thread_required: signal cleanup unavailable"
        )

    def terminate(number: int, _frame: FrameType | None) -> None:
        for sig in TERMINATION_SIGNALS:
            signal.signal(sig, signal.SIG_IGN)
        raise SystemExit(128 + number)

    previous = {sig: signal.getsignal(sig) for sig in TERMINATION_SIGNALS}
    try:
        for sig in previous:
            signal.signal(sig, terminate)
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)

# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Unwind owned analysis resources when an isolated worker is terminated."""

from __future__ import annotations

import signal
from collections.abc import Iterator
from contextlib import contextmanager
from math import isfinite
from types import FrameType


class AnalysisDeadline(BaseException):
    """A scheduler interruption must not become an unavailable mapper estimate."""


def _deadline(signum: int, frame: FrameType | None) -> None:
    del signum, frame
    raise AnalysisDeadline


@contextmanager
def analysis_deadline(seconds: float) -> Iterator[None]:
    """Bound one main-thread analysis; do not override an active caller timer."""
    if not isfinite(seconds) or seconds <= 0:
        raise ValueError("analysis deadline must be finite and positive")
    if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
        raise ValueError("analysis cannot replace an active caller timer")
    previous = signal.signal(signal.SIGALRM, _deadline)
    try:
        signal.setitimer(signal.ITIMER_REAL, seconds)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _terminate(signum: int, frame: FrameType | None) -> None:
    del frame
    raise SystemExit(128 + signum)


@contextmanager
def cleanup_on_termination() -> Iterator[None]:
    """Install in a worker's main thread; preserve the caller's signal policy.

    SIGTERM must unwind mapper and temporary-directory contexts. SIGKILL and
    tools that deliberately detach from their owned session cannot be unwound.
    """
    previous = signal.signal(signal.SIGTERM, _terminate)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)

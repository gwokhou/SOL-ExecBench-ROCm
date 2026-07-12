# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Standard-stream helpers for process execution."""

from __future__ import annotations

import sys


def flush_stdio_streams() -> None:
    """Flush standard streams while tolerating unavailable stream backends."""

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass

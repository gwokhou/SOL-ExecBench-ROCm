# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Private contracts shared by prediction components."""

from __future__ import annotations

_F16_WMMA_FLOPS = 2.0 * 16.0 * 16.0 * 16.0
_DEFAULT_WAVE_SIZE = 32.0
_FP32_BYTES = 4.0


class _PredictionUnavailableError(Exception):
    """Internal fail-closed signal carrying stable reasons."""

    def __init__(self, *reasons: str) -> None:
        super().__init__(", ".join(reasons))
        self.reasons = list(reasons)

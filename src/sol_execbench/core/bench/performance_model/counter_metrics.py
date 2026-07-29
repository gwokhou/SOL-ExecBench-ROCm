# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Unit-aware derived metrics from normalized rocprof counter values."""

from __future__ import annotations

from collections.abc import Mapping

_MEMORY_COUNTER_MULTIPLIERS = {
    "FETCH_SIZE": 1.0,
    "WRITE_SIZE": 1.0,
    "GL2C_EA_RDREQ_32B_SUM": 32.0,
    "GL2C_EA_RDREQ_64B_SUM": 64.0,
    "GL2C_EA_RDREQ_128B_SUM": 128.0,
    "GL2C_EA_WRREQ_64B_SUM": 64.0,
    "TCC_EA_RDREQ_32B_SUM": 32.0,
    "TCC_EA_WRREQ_64B_SUM": 64.0,
}


def counter_memory_bytes(counters: Mapping[str, float]) -> float:
    """Return traffic bytes from canonical byte and transaction counters."""
    return sum(
        counters.get(name, 0.0) * multiplier
        for name, multiplier in _MEMORY_COUNTER_MULTIPLIERS.items()
    )


__all__ = ["counter_memory_bytes"]

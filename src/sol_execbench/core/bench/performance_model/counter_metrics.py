# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Unit-aware derived metrics from normalized rocprof counter values."""

from __future__ import annotations

from collections.abc import Mapping

_MEMORY_COUNTER_MULTIPLIERS = {
    "FETCH_SIZE": 1024.0,
    "WRITE_SIZE": 1024.0,
    "GL2C_EA_RDREQ_32B_SUM": 32.0,
    "GL2C_EA_RDREQ_64B_SUM": 64.0,
    "GL2C_EA_RDREQ_128B_SUM": 128.0,
    "GL2C_EA_WRREQ_64B_SUM": 64.0,
    "TCC_EA_RDREQ_32B_SUM": 32.0,
    "TCC_EA_WRREQ_64B_SUM": 64.0,
}
_CACHE_REQUEST_COUNTERS = (
    ("GL2C_HIT_SUM", "GL2C_MISS_SUM"),
    ("TCC_HIT", "TCC_MISS"),
    ("L2CACHEHIT", "L2CACHEMISS"),
)
# AMD's RDNA system-SOL model defines one GL2 request as 128 bytes. These
# counters are used only when the direct external-traffic byte counters are
# present but zero, as happens for warm-cache replay.
_RDNA_GL2_REQUEST_BYTES = 128.0


def counter_memory_bytes(counters: Mapping[str, float]) -> float:
    """Return traffic bytes from canonical byte and transaction counters."""
    direct = sum(
        counters.get(name, 0.0)
        * (1.0 if name in {"FETCH_SIZE", "WRITE_SIZE"} else multiplier)
        for name, multiplier in _MEMORY_COUNTER_MULTIPLIERS.items()
    )
    if direct > 0:
        return direct
    for hit_name, miss_name in _CACHE_REQUEST_COUNTERS:
        if hit_name in counters or miss_name in counters:
            return (
                counters.get(hit_name, 0.0) + counters.get(miss_name, 0.0)
            ) * _RDNA_GL2_REQUEST_BYTES
    return 0.0


def counter_native_multiplier(name: str) -> float | None:
    """Return the declared conversion from native counter value to bytes."""
    return _MEMORY_COUNTER_MULTIPLIERS.get(name)


__all__ = ["counter_memory_bytes", "counter_native_multiplier"]

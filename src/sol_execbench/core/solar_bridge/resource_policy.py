# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Resource-policy queries across the SOL ExecBench to SOLAR boundary."""

from solar.analysis.orojenesis.configuration import (
    available_logical_cpu_count,
    orojenesis_mapper_thread_count,
)


def available_formal_mapper_logical_cpu_count() -> int | None:
    """Return the process-visible logical CPU budget used by the mapper."""
    return available_logical_cpu_count()


def formal_mapper_thread_count() -> int:
    """Return the canonical thread demand of one formal mapper invocation."""
    return orojenesis_mapper_thread_count()


__all__ = [
    "available_formal_mapper_logical_cpu_count",
    "formal_mapper_thread_count",
]

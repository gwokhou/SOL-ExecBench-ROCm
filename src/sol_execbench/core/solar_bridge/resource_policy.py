# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Resource-policy queries across the SOL ExecBench to SOLAR boundary."""

from solar.analysis.orojenesis.configuration import OROJENESIS_MAPPER_THREADS


def formal_mapper_thread_count() -> int:
    """Return the canonical thread demand of one formal mapper invocation."""
    return OROJENESIS_MAPPER_THREADS


__all__ = ["formal_mapper_thread_count"]

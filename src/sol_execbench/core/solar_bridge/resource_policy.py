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


def formal_mapper_job_limit(
    logical_cpus: int,
    *,
    mapper_threads: int,
) -> tuple[int, int]:
    """Return the safe job cap and unused complete-slot CPU remainder."""
    if logical_cpus <= 0:
        raise ValueError("available logical CPUs must be positive")
    if mapper_threads <= 0:
        raise ValueError("formal mapper threads must be positive")
    complete_slots, remaining_cpus = divmod(logical_cpus, mapper_threads)
    return max(1, complete_slots), remaining_cpus


def validate_formal_job_count(jobs: int, *, caller: str) -> None:
    """Reject job counts the visible CPU budget cannot safely carry.

    L4 spec: one shared bound for every SOLAR batch path (release runner
    and qualification CLI alike). jobs=1 always passes; jobs above the
    complete-slot mapper limit fails with the measured budget reason.
    """
    if jobs <= 0:
        raise ValueError(f"{caller} jobs must be positive")
    if jobs == 1:
        return
    logical_cpus = available_formal_mapper_logical_cpu_count()
    if logical_cpus is None:
        raise ValueError(
            f"{caller} cannot safely run jobs above 1 because available "
            "logical CPUs could not be detected",
        )
    mapper_threads = formal_mapper_thread_count()
    maximum, remaining_cpus = formal_mapper_job_limit(
        logical_cpus,
        mapper_threads=mapper_threads,
    )
    if jobs > maximum:
        remainder = (
            f", leaving {remaining_cpus} logical CPUs outside complete "
            "mapper slots"
            if remaining_cpus
            else ""
        )
        raise ValueError(
            f"{caller} jobs {jobs} exceed the safe limit {maximum}: "
            f"{logical_cpus} available logical CPUs / "
            f"{mapper_threads} mapper threads per workload{remainder}",
        )


__all__ = [
    "available_formal_mapper_logical_cpu_count",
    "formal_mapper_job_limit",
    "formal_mapper_thread_count",
    "validate_formal_job_count",
]

# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Conservative compile concurrency from affinity and visible cgroup v2 limits."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_COMPILE_JOBS = 1
_MEMORY_PER_JOB = 1024**3
_MEMORY_RESERVE = 512 * 1024**2
_CGROUP_ROOT = Path("/sys/fs/cgroup")
_CGROUP_MEMBERSHIP = Path("/proc/self/cgroup")


def _cgroup_directories() -> tuple[Path, ...]:
    """Inspect visible ancestors as well as a container's namespace root."""
    paths = [_CGROUP_ROOT]
    try:
        lines = _CGROUP_MEMBERSHIP.read_text().splitlines()
    except OSError:
        return tuple(paths)
    for line in lines:
        if line.startswith("0::"):
            relative = Path(line[3:].lstrip("/"))
            if ".." not in relative.parts:
                child = _CGROUP_ROOT / relative
                paths.extend(
                    p
                    for p in (child, *child.parents)
                    if p.is_relative_to(_CGROUP_ROOT)
                )
    return tuple(dict.fromkeys(paths))


def compile_job_limit(requested: int = DEFAULT_COMPILE_JOBS) -> int:
    """Cap jobs; absent/malformed resource evidence conservatively means one.

    One GiB per compiler plus 512 MiB coordinator reserve is a scheduling
    budget, not a guarantee of compiler peak RSS. The caller's hard cgroup
    memory/CPU limits still apply. cgroup v1/unknown layouts stay serial.
    """
    if (
        not isinstance(requested, int)
        or isinstance(requested, bool)
        or requested <= 0
    ):
        raise ValueError("compile jobs must be a positive integer")
    limits = [requested, len(os.sched_getaffinity(0))]
    cpu_known = memory_known = False
    for root in _cgroup_directories():
        try:
            quota, period = (root / "cpu.max").read_text().split()
            interval = int(period)
            if interval <= 0:
                raise ValueError("invalid CPU period")
            if quota != "max":
                limits.append(max(1, int(quota) // interval))
            cpu_known = True
            memory = (root / "memory.max").read_text().strip()
            if memory != "max":
                limits.append(
                    max(1, (int(memory) - _MEMORY_RESERVE) // _MEMORY_PER_JOB)
                )
            memory_known = True
        except OSError, ValueError:
            limits.append(1)
    if not cpu_known or not memory_known:
        limits.append(1)
    return max(1, min(limits))

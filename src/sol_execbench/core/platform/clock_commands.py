# SPDX-License-Identifier: Apache-2.0
"""Canonical AMD-SMI clock command contract shared by execution and sudoers."""

import os
import re
from enum import StrEnum

ENV_HOST_CLOCK_GPU_UUID = "SOL_EXECBENCH_HOST_CLOCK_GPU_UUID"
ENV_HOST_AMD_SMI = "SOL_EXECBENCH_HOST_AMD_SMI"
DEFAULT_HOST_AMD_SMI = "/usr/bin/amd-smi"

_UUID_PATTERN = re.compile(
    r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}"
)


class ClockLevel(StrEnum):
    """The only policies managed by the owned clock lifecycle."""

    AUTO = "AUTO"
    PEAK = "STABLE_PEAK"


def clock_executable(explicit: str | None = None) -> str:
    """Preserve legacy/container discovery or an explicit invocation path."""
    if explicit is not None:
        return explicit
    from sol_execbench.core.platform.runtime import resolve_rocm_tool_command

    return resolve_rocm_tool_command("amd-smi")


def host_clock_executable(explicit: str | None = None) -> str:
    """Select an explicit host path or stable default, never a PATH lookup."""
    selected = (
        explicit
        if explicit is not None
        else os.environ.get(ENV_HOST_AMD_SMI, DEFAULT_HOST_AMD_SMI)
    )
    if not os.path.isabs(selected) or "\0" in selected:
        raise ValueError("host AMD-SMI must be a nonempty absolute path")
    return selected


def clock_arguments(
    level: ClockLevel, gpu_uuid: str | None = None
) -> list[str]:
    """Build exact arguments; None explicitly retains legacy all-device scope."""
    level = ClockLevel(level)
    arguments = ["set", "-l", level.value]
    if gpu_uuid is not None:
        if _UUID_PATTERN.fullmatch(gpu_uuid) is None:
            raise ValueError("invalid AMD-SMI GPU UUID")
        arguments.extend(("-g", gpu_uuid))
    return arguments


def clock_command(
    executable: str, level: ClockLevel, gpu_uuid: str | None = None
) -> list[str]:
    """Bind the executable to the canonical command arguments."""
    return [executable, *clock_arguments(level, gpu_uuid)]

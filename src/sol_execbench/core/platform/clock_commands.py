# SPDX-License-Identifier: Apache-2.0
"""Canonical AMD-SMI clock command contract shared by execution and sudoers."""

import re
from enum import StrEnum

_UUID_PATTERN = re.compile(
    r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}"
)


class ClockLevel(StrEnum):
    """The only policies managed by the owned clock lifecycle."""

    AUTO = "AUTO"
    PEAK = "STABLE_PEAK"


def clock_executable(explicit: str | None = None) -> str:
    """Resolve the same executable spelling for authorization and execution."""
    if explicit is not None:
        return explicit
    from sol_execbench.core.platform.runtime import resolve_rocm_tool_command

    return resolve_rocm_tool_command("amd-smi")


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

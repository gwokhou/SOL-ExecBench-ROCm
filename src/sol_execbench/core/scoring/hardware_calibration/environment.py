# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Runtime GPU discovery and architecture-specific calibration declarations."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from typing import Protocol

from sol_execbench.core.scoring.hardware_calibration.hip_probe import (
    CalibrationProfileKey,
)


@dataclass(frozen=True)
class GpuEnvironment:
    device: int
    architecture: str
    name: str | None = None


class GpuRuntime(Protocol):
    def architecture_for(self, device: int) -> str: ...


class RocmInfoRuntime:
    """Discover the visible GPU architecture without importing HIP Python bindings."""

    def architecture_for(self, device: int) -> str:
        try:
            output = subprocess.check_output(
                ("rocminfo",), text=True, stderr=subprocess.DEVNULL
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError("HIP runtime discovery unavailable") from exc
        architectures = re.findall(r"(?:Name|Marketing Name):\s*(gfx[0-9a-z]+)", output)
        try:
            return architectures[device].lower()
        except IndexError as exc:
            raise RuntimeError(f"HIP device {device} was not discovered") from exc


def discover_gpu(device: int, runtime: GpuRuntime | None = None) -> GpuEnvironment:
    """Return runtime evidence for one visible HIP device."""
    if device < 0:
        raise ValueError("device must be non-negative")
    runtime = runtime or RocmInfoRuntime()
    return GpuEnvironment(
        device=device, architecture=runtime.architecture_for(device).lower()
    )


@dataclass(frozen=True)
class ArchitectureAdapter:
    family: str
    candidates: tuple[CalibrationProfileKey, ...]
    supports_clock_lock: bool = False


def _adapter(family: str, *, supports_clock_lock: bool = False) -> ArchitectureAdapter:
    return ArchitectureAdapter(
        family=family,
        supports_clock_lock=supports_clock_lock,
        candidates=(
            CalibrationProfileKey("compute", "vector", "fp32", "fp32", "portable"),
            CalibrationProfileKey("memory", "stream_copy", "fp32", "fp32", "portable"),
            CalibrationProfileKey("compute", "vector", "fp32", "fp32", family),
            CalibrationProfileKey("memory", "stream_copy", "fp32", "fp32", family),
        ),
    )


_ADAPTERS = {
    "gfx12": _adapter("gfx12", supports_clock_lock=True),
    "gfx94": _adapter("gfx94"),
    "gfx95": _adapter("gfx95"),
}


def adapter_for(architecture: str) -> ArchitectureAdapter:
    """Return the declared calibration adapter for a supported AMD ISA family."""
    normalized = architecture.lower()
    for prefix, adapter in _ADAPTERS.items():
        if normalized.startswith(prefix):
            return adapter
    raise ValueError(f"unsupported GPU architecture: {architecture}")

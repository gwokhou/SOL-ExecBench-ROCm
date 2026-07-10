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
    uuid: str | None = None
    rocm_version: str | None = None


class GpuRuntime(Protocol):
    def architecture_for(self, device: int) -> str: ...

    def uuid_for(self, device: int) -> str | None: ...

    def rocm_version(self) -> str | None: ...


class RocmInfoRuntime:
    """Discover the visible GPU architecture without importing HIP Python bindings."""

    def architecture_for(self, device: int) -> str:
        return self._fields_for(device)[0]

    def _fields_for(self, device: int) -> tuple[str, str | None]:
        try:
            output = subprocess.check_output(
                ("rocminfo",), text=True, stderr=subprocess.DEVNULL
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError("HIP runtime discovery unavailable") from exc
        architectures = re.findall(r"(?:Name|Marketing Name):\s*(gfx[0-9a-z]+)", output)
        uuids = re.findall(r"(?:Uuid|UUID):\s*([^\s]+)", output, re.IGNORECASE)
        try:
            return architectures[device].lower(), (
                uuids[device] if device < len(uuids) else None
            )
        except IndexError as exc:
            raise RuntimeError(f"HIP device {device} was not discovered") from exc

    def uuid_for(self, device: int) -> str | None:
        return self._fields_for(device)[1]

    def rocm_version(self) -> str | None:
        try:
            output = subprocess.check_output(
                ("hipcc", "--version"), text=True, stderr=subprocess.DEVNULL
            )
        except (OSError, subprocess.SubprocessError):
            return None
        match = re.search(r"(?:HIP|ROCm) version:\s*([^\s]+)", output, re.I)
        return match.group(1) if match else None


def discover_gpu(device: int, runtime: GpuRuntime | None = None) -> GpuEnvironment:
    """Return runtime evidence for one visible HIP device."""
    if device < 0:
        raise ValueError("device must be non-negative")
    runtime = runtime or RocmInfoRuntime()
    uuid_for = getattr(runtime, "uuid_for", lambda _: None)
    rocm_version = getattr(runtime, "rocm_version", lambda: None)
    return GpuEnvironment(
        device=device,
        architecture=runtime.architecture_for(device).lower(),
        uuid=uuid_for(device),
        rocm_version=rocm_version(),
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
            # These are declarations, not claims: the HIP probe marks a path
            # unavailable or unknown if the active toolchain cannot execute it.
            CalibrationProfileKey(
                "compute", "matrix", "bf16", "bf16", f"{family}_mfma"
            ),
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

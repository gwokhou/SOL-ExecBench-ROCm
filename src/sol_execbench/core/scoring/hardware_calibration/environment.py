# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Runtime GPU discovery and architecture-specific calibration declarations."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from typing import Protocol

from sol_execbench.core.platform.runtime import resolve_rocm_tool
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
                (_rocm_command("rocminfo"),), text=True, stderr=subprocess.DEVNULL
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError("HIP runtime discovery unavailable") from exc
        gpu_agents = _gpu_agents_from_rocminfo(output)
        try:
            return gpu_agents[device]
        except IndexError as exc:
            raise RuntimeError(f"HIP device {device} was not discovered") from exc

    def uuid_for(self, device: int) -> str | None:
        return self._fields_for(device)[1]

    def rocm_version(self) -> str | None:
        try:
            output = subprocess.check_output(
                (_rocm_command("hipcc"), "--version"),
                text=True,
                stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        match = re.search(r"(?:HIP|ROCm) version:\s*([^\s]+)", output, re.I)
        return match.group(1) if match else None


def _rocm_command(tool: str) -> str:
    """Use the discovered absolute ROCm tool path when available."""
    path = resolve_rocm_tool(tool)
    return str(path) if path is not None else tool


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


def _gpu_agents_from_rocminfo(output: str) -> list[tuple[str, str | None]]:
    """Return GPU architecture/UUID pairs from the same ROCr agent block.

    ``rocminfo`` lists CPU agents before GPU agents on common desktop systems.
    Collecting architectures and UUIDs independently incorrectly pairs gfx1200
    with ``CPU-XX``; authority evidence must bind both values to one GPU agent.
    """
    agents = re.split(r"\*+\s*\n\s*Agent\s+\d+\s*\n\s*\*+", output)
    result: list[tuple[str, str | None]] = []
    for agent in agents[1:]:
        architecture = re.search(r"^\s*Name:\s*(gfx[0-9a-z]+)\s*$", agent, re.M)
        if architecture is None:
            continue
        uuid = re.search(r"^\s*Uuid:\s*([^\s]+)", agent, re.M | re.I)
        result.append(
            (architecture.group(1).lower(), uuid.group(1) if uuid is not None else None)
        )
    return result


@dataclass(frozen=True)
class ArchitectureAdapter:
    family: str
    candidates: tuple[CalibrationProfileKey, ...]
    diagnostic_candidates: tuple[CalibrationProfileKey, ...] = ()
    supports_clock_lock: bool = False

    @property
    def all_candidates(self) -> tuple[CalibrationProfileKey, ...]:
        """Return core and opt-in diagnostic profiles without duplication."""
        return self.candidates + self.diagnostic_candidates


def _matrix_key(family: str, dtype: str = "bf16") -> CalibrationProfileKey:
    path = "wmma" if family == "gfx12" else "mfma"
    return CalibrationProfileKey("compute", "matrix", dtype, dtype, path)


def _adapter(family: str, *, supports_clock_lock: bool = False) -> ArchitectureAdapter:
    candidates = [
        CalibrationProfileKey("compute", "vector", "fp32", "fp32", "portable"),
        CalibrationProfileKey("memory", "stream_copy", "fp32", "fp32", "portable"),
        CalibrationProfileKey("compute", "vector", "fp32", "fp32", family),
        _matrix_key(family),
        CalibrationProfileKey("memory", "stream_copy", "fp32", "fp32", family),
    ]
    diagnostic_candidates: list[CalibrationProfileKey] = []
    if family == "gfx12":
        diagnostic_candidates.append(
            CalibrationProfileKey("compute", "matrix", "fp32", "fp32", family)
        )
        diagnostic_candidates.append(
            CalibrationProfileKey("compute", "vector", "bf16", "bf16", family)
        )
        diagnostic_candidates.append(
            CalibrationProfileKey("compute", "vector", "fp16", "fp16", family)
        )
        candidates.append(_matrix_key(family, "fp16"))
        diagnostic_candidates.append(
            CalibrationProfileKey("memory", "stream_copy", "bf16", "bf16", family)
        )
        diagnostic_candidates.extend(
            (
                CalibrationProfileKey("memory", "stream_copy", "bf16", "fp32", family),
                CalibrationProfileKey("memory", "stream_copy", "fp32", "bf16", family),
            )
        )
        candidates.append(
            CalibrationProfileKey("memory", "stream_copy", "fp16", "fp16", "portable")
        )
        candidates.append(
            CalibrationProfileKey("memory", "stream_copy", "fp16", "fp16", family)
        )
        diagnostic_candidates.append(
            CalibrationProfileKey("compute", "reduction", "fp32", "fp32", family)
        )
        diagnostic_candidates.append(
            CalibrationProfileKey("compute", "transcendental", "fp32", "fp32", family)
        )
    return ArchitectureAdapter(
        family=family,
        supports_clock_lock=supports_clock_lock,
        candidates=tuple(candidates),
        diagnostic_candidates=tuple(diagnostic_candidates),
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

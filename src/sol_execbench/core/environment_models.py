# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Environment evidence models and injectable probe types."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import ConfigDict, Field

from .data.base_model import BaseModelWithDocstrings


ENVIRONMENT_SNAPSHOT_SCHEMA_VERSION = "sol_execbench.environment_snapshot.v1"
DEFAULT_PROBE_TIMEOUT_SECONDS = 3.0


class EnvironmentEvidenceStatus(str, Enum):
    """Status vocabulary for optional environment evidence collection."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


class ToolProbeResult(BaseModelWithDocstrings):
    """Captured result from one environment-probing command."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    tool: str
    """Logical tool name, such as ``rocminfo``."""
    command: list[str] = Field(default_factory=list)
    """Command attempted for the probe."""
    path: str | None = None
    """Resolved executable path when available."""
    status: EnvironmentEvidenceStatus
    """Probe status."""
    returncode: int | None = None
    """Process return code for executed probes."""
    stdout_tail: str = ""
    """Bounded stdout tail for diagnostics."""
    stderr_tail: str = ""
    """Bounded stderr tail for diagnostics."""
    timeout_seconds: float | None = None
    """Timeout applied to the probe."""
    parsed: dict[str, Any] = Field(default_factory=dict)
    """Conservative parsed fields derived from probe output."""


class GpuEnvironmentSummary(BaseModelWithDocstrings):
    """Best-effort summary for one detected AMD GPU agent."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    source: str
    """Probe source that produced this summary."""
    index: int | None = None
    """Device index when known."""
    name: str | None = None
    """Marketing or runtime device name when known."""
    gfx_target: str | None = None
    """AMD gfx architecture target such as ``gfx1200`` or ``gfx942``."""


class RocmEnvironmentSummary(BaseModelWithDocstrings):
    """Best-effort ROCm runtime and visibility summary."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    version: str | None = None
    """ROCm version when detected."""
    hip_visible_devices: str | None = None
    """Value of ``HIP_VISIBLE_DEVICES``."""
    rocr_visible_devices: str | None = None
    """Value of ``ROCR_VISIBLE_DEVICES``."""
    hsa_override_gfx_version: str | None = None
    """Value of ``HSA_OVERRIDE_GFX_VERSION``."""


class PytorchRocmSummary(BaseModelWithDocstrings):
    """Best-effort PyTorch ROCm environment summary."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    available: bool
    """Whether PyTorch ROCm reports an available device."""
    torch_version: str | None = None
    """Imported PyTorch version."""
    hip_version: str | None = None
    """``torch.version.hip`` value."""
    cuda_version: str | None = None
    """``torch.version.cuda`` value; should normally be ``None`` on ROCm."""
    device_count: int | None = None
    """Visible PyTorch CUDA/HIP device count."""
    device_name: str | None = None
    """Name of device 0 when available."""
    gfx_target: str | None = None
    """gfx target for device 0 when available."""
    error: str | None = None
    """Import or runtime error captured while probing PyTorch."""


class EnvironmentSnapshot(BaseModelWithDocstrings):
    """Optional ROCm environment evidence snapshot."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    schema_version: str = ENVIRONMENT_SNAPSHOT_SCHEMA_VERSION
    """Environment snapshot schema version."""
    generated_at: str
    """UTC timestamp when the snapshot was generated."""
    collection_status: EnvironmentEvidenceStatus
    """Aggregate status for the snapshot."""
    tools: dict[str, ToolProbeResult] = Field(default_factory=dict)
    """Per-tool probe evidence."""
    gpus: list[GpuEnvironmentSummary] = Field(default_factory=list)
    """Best-effort detected GPU summaries."""
    rocm: RocmEnvironmentSummary = Field(default_factory=RocmEnvironmentSummary)
    """ROCm runtime visibility summary."""
    pytorch: PytorchRocmSummary | None = None
    """PyTorch ROCm summary when collection was attempted."""
    visible_devices: dict[str, str] = Field(default_factory=dict)
    """Relevant device-visibility environment variables."""
    warnings: list[str] = Field(default_factory=list)
    """Non-fatal collection warnings."""


class EnvironmentCheckResult(BaseModelWithDocstrings):
    """One diagnostic preflight check result."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    name: str
    """Stable check name."""
    status: EnvironmentEvidenceStatus
    """Check status."""
    message: str
    """Human-readable result summary."""
    remediation: str | None = None
    """Optional remediation hint."""


class EnvironmentDiagnostics(BaseModelWithDocstrings):
    """Standalone environment diagnostic payload."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    schema_version: str = "sol_execbench.environment_diagnostics.v1"
    """Diagnostics schema version."""
    generated_at: str
    """UTC timestamp when diagnostics were generated."""
    status: EnvironmentEvidenceStatus
    """Aggregate diagnostic status."""
    snapshot: EnvironmentSnapshot
    """Environment snapshot evidence."""
    checks: list[EnvironmentCheckResult] = Field(default_factory=list)
    """Preflight check results."""


@dataclass(frozen=True)
class ProbeCompletedProcess:
    """Small subprocess result shape used by injectable probe runners."""

    returncode: int
    stdout: str = ""
    stderr: str = ""


ProbeRunner = Callable[[list[str], float], ProbeCompletedProcess]
Which = Callable[[str], str | None]

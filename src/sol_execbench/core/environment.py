# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Optional ROCm runtime environment snapshot evidence.

This module is deliberately separate from the canonical trace ``Environment``
model. Snapshot evidence is collected explicitly and can be attached to richer
run metadata by callers without changing trace JSONL semantics.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import ConfigDict, Field

from .data.base_model import BaseModelWithDocstrings
from .text_utils import text_tail as _tail


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


def collect_environment_snapshot(
    *,
    runner: ProbeRunner | None = None,
    which: Which = shutil.which,
    timeout_seconds: float = DEFAULT_PROBE_TIMEOUT_SECONDS,
    collect_pytorch: bool = True,
    now: Callable[[], datetime] | None = None,
) -> EnvironmentSnapshot:
    """Collect optional ROCm environment evidence.

    The function performs no work until called. All external command execution
    is timeout-bounded and injectable for tests.
    """

    effective_runner = runner or _run_probe
    generated_at = (now or (lambda: datetime.now(UTC)))().isoformat()
    tools = {
        "amd-smi": probe_tool(
            "amd-smi",
            ["amd-smi", "static", "-a"],
            runner=effective_runner,
            which=which,
            timeout_seconds=timeout_seconds,
        ),
        "rocminfo": probe_tool(
            "rocminfo",
            ["rocminfo"],
            runner=effective_runner,
            which=which,
            timeout_seconds=timeout_seconds,
        ),
        "rocm_agent_enumerator": probe_tool(
            "rocm_agent_enumerator",
            ["rocm_agent_enumerator"],
            runner=effective_runner,
            which=which,
            timeout_seconds=timeout_seconds,
        ),
    }
    pytorch = collect_pytorch_rocm_summary() if collect_pytorch else None
    visible_devices = _visible_device_environment()
    gpus = _summarize_gpus(tools, pytorch)
    warnings = _snapshot_warnings(tools, pytorch)
    return EnvironmentSnapshot(
        generated_at=generated_at,
        collection_status=_aggregate_status(tools, pytorch),
        tools=tools,
        gpus=gpus,
        rocm=RocmEnvironmentSummary(
            hip_visible_devices=visible_devices.get("HIP_VISIBLE_DEVICES"),
            rocr_visible_devices=visible_devices.get("ROCR_VISIBLE_DEVICES"),
            hsa_override_gfx_version=visible_devices.get("HSA_OVERRIDE_GFX_VERSION"),
        ),
        pytorch=pytorch,
        visible_devices=visible_devices,
        warnings=warnings,
    )


def build_environment_diagnostics(
    *,
    snapshot_collector: Callable[
        [], EnvironmentSnapshot
    ] = collect_environment_snapshot,
    smoke_checker: Callable[[], list[EnvironmentCheckResult]] | None = None,
    now: Callable[[], datetime] | None = None,
) -> EnvironmentDiagnostics:
    """Build standalone environment diagnostics for ``sol-execbench doctor``."""

    snapshot = snapshot_collector()
    checks = _tool_checks(snapshot)
    checks.extend(
        smoke_checker() if smoke_checker is not None else run_pytorch_smoke_checks()
    )
    status = _aggregate_check_status(
        [snapshot.collection_status, *(check.status for check in checks)]
    )
    generated_at = (now or (lambda: datetime.now(UTC)))().isoformat()
    return EnvironmentDiagnostics(
        generated_at=generated_at,
        status=status,
        snapshot=snapshot,
        checks=checks,
    )


def run_pytorch_smoke_checks() -> list[EnvironmentCheckResult]:
    """Run lightweight PyTorch ROCm smoke checks for diagnostics only."""

    try:
        import torch
    except ImportError as exc:
        return [
            EnvironmentCheckResult(
                name="pytorch_rocm_import",
                status=EnvironmentEvidenceStatus.UNAVAILABLE,
                message=f"PyTorch unavailable: {exc}",
                remediation="Install the PyTorch ROCm wheel for this environment.",
            )
        ]

    hip_version = getattr(getattr(torch, "version", None), "hip", None)
    try:
        available = bool(torch.cuda.is_available()) and hip_version is not None
    except RuntimeError as exc:
        return [
            EnvironmentCheckResult(
                name="pytorch_rocm_runtime",
                status=EnvironmentEvidenceStatus.FAILED,
                message=f"PyTorch ROCm runtime failed: {exc}",
            )
        ]

    if not available:
        return [
            EnvironmentCheckResult(
                name="pytorch_rocm_runtime",
                status=EnvironmentEvidenceStatus.UNAVAILABLE,
                message="PyTorch ROCm device is not available",
                remediation=(
                    "Check ROCm installation, device permissions, and "
                    "HIP_VISIBLE_DEVICES."
                ),
            ),
            EnvironmentCheckResult(
                name="device_memory_copy",
                status=EnvironmentEvidenceStatus.SKIPPED,
                message="Skipped because PyTorch ROCm runtime is unavailable",
            ),
            EnvironmentCheckResult(
                name="event_timing",
                status=EnvironmentEvidenceStatus.SKIPPED,
                message="Skipped because PyTorch ROCm runtime is unavailable",
            ),
        ]

    checks = [
        EnvironmentCheckResult(
            name="pytorch_rocm_runtime",
            status=EnvironmentEvidenceStatus.AVAILABLE,
            message="PyTorch ROCm runtime is available",
        )
    ]
    try:
        tensor = torch.ones((1,), device="cuda")
        copied = tensor.to("cpu")
        if float(copied.item()) == 1.0:
            checks.append(
                EnvironmentCheckResult(
                    name="device_memory_copy",
                    status=EnvironmentEvidenceStatus.AVAILABLE,
                    message="Device memory copy succeeded",
                )
            )
        else:
            checks.append(
                EnvironmentCheckResult(
                    name="device_memory_copy",
                    status=EnvironmentEvidenceStatus.FAILED,
                    message="Device memory copy returned an unexpected value",
                )
            )
    except Exception as exc:
        checks.append(
            EnvironmentCheckResult(
                name="device_memory_copy",
                status=EnvironmentEvidenceStatus.FAILED,
                message=f"Device memory copy failed: {exc}",
            )
        )

    try:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        end.record()
        torch.cuda.synchronize()
        _ = start.elapsed_time(end)
        checks.append(
            EnvironmentCheckResult(
                name="event_timing",
                status=EnvironmentEvidenceStatus.AVAILABLE,
                message="HIP-backed PyTorch event timing succeeded",
            )
        )
    except Exception as exc:
        checks.append(
            EnvironmentCheckResult(
                name="event_timing",
                status=EnvironmentEvidenceStatus.FAILED,
                message=f"Event timing failed: {exc}",
            )
        )
    return checks


def probe_tool(
    tool: str,
    command: list[str],
    *,
    runner: ProbeRunner,
    which: Which = shutil.which,
    timeout_seconds: float = DEFAULT_PROBE_TIMEOUT_SECONDS,
) -> ToolProbeResult:
    """Run one bounded environment probe."""

    path = which(tool)
    if path is None:
        return ToolProbeResult(
            tool=tool,
            command=command,
            status=EnvironmentEvidenceStatus.UNAVAILABLE,
            timeout_seconds=timeout_seconds,
        )
    try:
        completed = runner(command, timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        return ToolProbeResult(
            tool=tool,
            command=command,
            path=path,
            status=EnvironmentEvidenceStatus.TIMEOUT,
            stdout_tail=_tail(exc.stdout),
            stderr_tail=_tail(exc.stderr),
            timeout_seconds=timeout_seconds,
        )
    except OSError as exc:
        return ToolProbeResult(
            tool=tool,
            command=command,
            path=path,
            status=EnvironmentEvidenceStatus.FAILED,
            stderr_tail=_tail(str(exc)),
            timeout_seconds=timeout_seconds,
        )

    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    return ToolProbeResult(
        tool=tool,
        command=command,
        path=path,
        status=(
            EnvironmentEvidenceStatus.AVAILABLE
            if completed.returncode == 0
            else EnvironmentEvidenceStatus.FAILED
        ),
        returncode=completed.returncode,
        stdout_tail=_tail(completed.stdout),
        stderr_tail=_tail(completed.stderr),
        timeout_seconds=timeout_seconds,
        parsed=_parse_probe_output(output),
    )


def collect_pytorch_rocm_summary() -> PytorchRocmSummary:
    """Collect PyTorch ROCm metadata without requiring PyTorch at import time."""

    try:
        import torch
    except ImportError as exc:
        return PytorchRocmSummary(available=False, error=str(exc))

    torch_version = str(getattr(torch, "__version__", ""))
    version = getattr(torch, "version", None)
    hip_version = getattr(version, "hip", None)
    cuda_version = getattr(version, "cuda", None)
    try:
        available = bool(torch.cuda.is_available()) and hip_version is not None
        device_count = int(torch.cuda.device_count()) if available else 0
        device_name = torch.cuda.get_device_name(0) if device_count else None
        gfx_target = None
        if device_count:
            props = torch.cuda.get_device_properties(0)
            raw_arch = getattr(props, "gcnArchName", "") or getattr(
                props, "gfx_arch_name", ""
            )
            gfx_target = str(raw_arch).split(":", maxsplit=1)[0] or None
        return PytorchRocmSummary(
            available=available,
            torch_version=torch_version,
            hip_version=hip_version,
            cuda_version=cuda_version,
            device_count=device_count,
            device_name=device_name,
            gfx_target=gfx_target,
        )
    except (RuntimeError, AttributeError) as exc:
        return PytorchRocmSummary(
            available=False,
            torch_version=torch_version,
            hip_version=hip_version,
            cuda_version=cuda_version,
            error=str(exc),
        )


def _run_probe(command: list[str], timeout_seconds: float) -> ProbeCompletedProcess:
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    return ProbeCompletedProcess(
        returncode=completed.returncode,
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
    )


def _aggregate_status(
    tools: dict[str, ToolProbeResult],
    pytorch: PytorchRocmSummary | None,
) -> EnvironmentEvidenceStatus:
    statuses = {tool.status for tool in tools.values()}
    if pytorch and pytorch.available:
        return EnvironmentEvidenceStatus.AVAILABLE
    if EnvironmentEvidenceStatus.AVAILABLE in statuses:
        return EnvironmentEvidenceStatus.AVAILABLE
    if EnvironmentEvidenceStatus.FAILED in statuses:
        return EnvironmentEvidenceStatus.FAILED
    if EnvironmentEvidenceStatus.TIMEOUT in statuses:
        return EnvironmentEvidenceStatus.TIMEOUT
    return EnvironmentEvidenceStatus.UNAVAILABLE


def _parse_probe_output(output: str) -> dict[str, Any]:
    gfx_targets = sorted(set(re.findall(r"\bgfx[0-9a-fA-F]+\b", output)))
    parsed: dict[str, Any] = {}
    if gfx_targets:
        parsed["gfx_targets"] = gfx_targets
    marketing_names = []
    for line in output.splitlines():
        if "Marketing Name" in line or "GPU" in line and "Name" in line:
            _, _, value = line.partition(":")
            value = value.strip()
            if value:
                marketing_names.append(value)
    if marketing_names:
        parsed["names"] = marketing_names[:8]
    return parsed


def _summarize_gpus(
    tools: dict[str, ToolProbeResult],
    pytorch: PytorchRocmSummary | None,
) -> list[GpuEnvironmentSummary]:
    gpus: list[GpuEnvironmentSummary] = []
    if pytorch and (pytorch.device_name or pytorch.gfx_target):
        gpus.append(
            GpuEnvironmentSummary(
                source="pytorch",
                index=0,
                name=pytorch.device_name,
                gfx_target=pytorch.gfx_target,
            )
        )
    for tool_name, result in tools.items():
        gfx_targets = result.parsed.get("gfx_targets")
        if isinstance(gfx_targets, list):
            for index, gfx_target in enumerate(gfx_targets):
                gpus.append(
                    GpuEnvironmentSummary(
                        source=tool_name,
                        index=index,
                        gfx_target=str(gfx_target),
                    )
                )
    return gpus


def _snapshot_warnings(
    tools: dict[str, ToolProbeResult],
    pytorch: PytorchRocmSummary | None,
) -> list[str]:
    warnings: list[str] = []
    for name, result in tools.items():
        if result.status != EnvironmentEvidenceStatus.AVAILABLE:
            warnings.append(f"{name}:{result.status.value}")
    if pytorch and not pytorch.available:
        warnings.append("pytorch_rocm:unavailable")
    return warnings


def _tool_checks(snapshot: EnvironmentSnapshot) -> list[EnvironmentCheckResult]:
    checks: list[EnvironmentCheckResult] = []
    for name, result in snapshot.tools.items():
        unavailable = result.status == EnvironmentEvidenceStatus.UNAVAILABLE
        checks.append(
            EnvironmentCheckResult(
                name=f"tool:{name}",
                status=result.status,
                message=(
                    f"{name} not found on PATH"
                    if unavailable
                    else f"{name} probe {result.status.value}"
                ),
                remediation=(
                    "Install ROCm tools or ensure /opt/rocm/bin is on PATH."
                    if unavailable
                    else None
                ),
            )
        )
    return checks


def _aggregate_check_status(
    statuses: list[EnvironmentEvidenceStatus],
) -> EnvironmentEvidenceStatus:
    if any(status == EnvironmentEvidenceStatus.FAILED for status in statuses):
        return EnvironmentEvidenceStatus.FAILED
    if any(status == EnvironmentEvidenceStatus.TIMEOUT for status in statuses):
        return EnvironmentEvidenceStatus.TIMEOUT
    if any(status == EnvironmentEvidenceStatus.AVAILABLE for status in statuses):
        return EnvironmentEvidenceStatus.AVAILABLE
    if any(status == EnvironmentEvidenceStatus.UNAVAILABLE for status in statuses):
        return EnvironmentEvidenceStatus.UNAVAILABLE
    return EnvironmentEvidenceStatus.SKIPPED


def _visible_device_environment() -> dict[str, str]:
    keys = ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "HSA_OVERRIDE_GFX_VERSION")
    return {key: value for key in keys if (value := os.environ.get(key)) is not None}

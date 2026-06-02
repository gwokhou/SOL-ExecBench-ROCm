# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""ROCm toolchain capability registry and routing helpers.

This module models tool availability and routing decisions. It does not collect
static kernel evidence or mutate canonical trace JSONL.
"""

from __future__ import annotations

import fnmatch
import shutil
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from enum import Enum

from pydantic import ConfigDict, Field

from .data.base_model import BaseModelWithDocstrings
from .environment import ProbeCompletedProcess


TOOLCHAIN_ROUTING_SCHEMA_VERSION = "sol_execbench.toolchain_routing.v1"
DEFAULT_TOOLCHAIN_PROBE_TIMEOUT_SECONDS = 3.0


class ToolLifecycle(str, Enum):
    """Lifecycle state for a ROCm-related tool entry."""

    ACTIVE = "active"
    DEPRECATED = "deprecated"
    MIGRATED = "migrated"
    PLANNED = "planned"
    REJECTED = "rejected"
    CANDIDATE = "candidate"


class ToolchainEvidenceLevel(str, Enum):
    """Evidence level a tool may support."""

    RUNTIME = "runtime"
    PROFILING = "profiling"
    STATIC = "static"
    DERIVED_SCORE = "derived_score"


class ToolchainArtifactType(str, Enum):
    """Artifact or workload class used for routing."""

    EXECUTABLE_RUN = "executable_run"
    ROCM_BINARY = "rocm_binary"
    ELF_OBJECT = "elf_object"
    HIP_COMPILER_OUTPUT = "hip_compiler_output"
    TRITON_ARTIFACT = "triton_artifact"
    STATIC_FUTURE = "static_future"
    NONE = "none"


class ToolchainStatus(str, Enum):
    """Status vocabulary for routing decisions."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED_ARCH = "unsupported_arch"
    UNSUPPORTED_ARTIFACT = "unsupported_artifact"
    DEPRECATED = "deprecated"
    MIGRATED = "migrated"
    PLANNED = "planned"
    REJECTED = "rejected"
    FAILED = "failed"


class ToolchainProbeResult(BaseModelWithDocstrings):
    """Bounded dynamic probe result for one tool."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    tool_id: str
    """Stable tool identifier."""
    command: list[str] = Field(default_factory=list)
    """Command attempted for the probe."""
    path: str | None = None
    """Resolved executable path when present."""
    status: ToolchainStatus
    """Probe status."""
    returncode: int | None = None
    """Process return code when a command was executed."""
    stdout_tail: str = ""
    """Bounded stdout tail."""
    stderr_tail: str = ""
    """Bounded stderr tail."""
    timeout_seconds: float | None = None
    """Probe timeout."""


class ToolchainCapability(BaseModelWithDocstrings):
    """One registry entry describing a ROCm-related tool capability."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    tool_id: str
    """Stable tool identifier."""
    display_name: str
    """Human-readable tool name."""
    lifecycle: ToolLifecycle
    """Tool lifecycle state."""
    replacement_tool_id: str | None = None
    """Replacement tool identifier for migrated or deprecated tools."""
    evidence_levels: list[ToolchainEvidenceLevel] = Field(default_factory=list)
    """Evidence levels this tool can contribute to."""
    artifact_types: list[ToolchainArtifactType] = Field(default_factory=list)
    """Artifact classes this tool can consume or produce evidence for."""
    hardware_generations: list[str] = Field(default_factory=list)
    """Hardware generations such as RDNA 4, CDNA 3, or CDNA 4."""
    gpu_arch_patterns: list[str] = Field(default_factory=list)
    """gfx architecture glob patterns supported by this registry entry."""
    rocm_version_min: str | None = None
    """Minimum ROCm version when known."""
    rocm_version_max: str | None = None
    """Maximum ROCm version when known."""
    expected_binaries: list[str] = Field(default_factory=list)
    """Executables expected on PATH for dynamic probing."""
    probe_command: list[str] = Field(default_factory=list)
    """Safe command used for bounded dynamic probing."""
    source_refs: list[str] = Field(default_factory=list)
    """Primary source references supporting this registry entry."""
    notes: str = ""
    """Boundary notes for the tool."""


class ToolchainRoutingRequest(BaseModelWithDocstrings):
    """Request to route a tool for a target evidence level and artifact."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    evidence_level: ToolchainEvidenceLevel
    """Evidence level requested."""
    artifact_type: ToolchainArtifactType = ToolchainArtifactType.NONE
    """Artifact class to route for."""
    gpu_architecture: str | None = None
    """Requested gfx architecture such as gfx1200 or gfx942."""
    hardware_generation: str | None = None
    """Requested hardware generation when known."""
    rocm_version: str | None = None
    """Requested ROCm version when known."""


class ToolchainRoutingDecision(BaseModelWithDocstrings):
    """One routing decision for a registry entry."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    tool_id: str
    """Tool that was considered."""
    status: ToolchainStatus
    """Routing status for this tool."""
    reason_code: str
    """Stable reason code."""
    reason: str
    """Human-readable reason."""
    selected: bool = False
    """Whether this tool was selected as the route."""
    fallback_tool_id: str | None = None
    """Fallback tool identifier when relevant."""
    lifecycle: ToolLifecycle
    """Lifecycle state from the registry."""
    source_refs: list[str] = Field(default_factory=list)
    """Source references supporting the decision."""
    probe: ToolchainProbeResult | None = None
    """Dynamic probe evidence when attempted."""


class ToolchainRoutingReport(BaseModelWithDocstrings):
    """Toolchain routing report for one request."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    schema_version: str = TOOLCHAIN_ROUTING_SCHEMA_VERSION
    """Routing schema version."""
    generated_at: str
    """UTC timestamp when the report was generated."""
    diagnostic_only: bool = True
    """Routing is diagnostic metadata only."""
    correctness_authority: bool = False
    """Routing never proves correctness."""
    performance_authority: bool = False
    """Routing never proves performance."""
    leaderboard_authority: bool = False
    """Routing never proves leaderboard readiness."""
    request: ToolchainRoutingRequest
    """Requested routing target."""
    selected_tool_id: str | None = None
    """Selected tool when one is available."""
    decisions: list[ToolchainRoutingDecision] = Field(default_factory=list)
    """All considered routing decisions."""


ProbeRunner = Callable[[list[str], float], ProbeCompletedProcess]
Which = Callable[[str], str | None]


def default_toolchain_registry() -> list[ToolchainCapability]:
    """Return the built-in ROCm toolchain capability registry."""

    return [
        ToolchainCapability(
            tool_id="rocprofv3",
            display_name="ROCprofiler SDK rocprofv3",
            lifecycle=ToolLifecycle.ACTIVE,
            evidence_levels=[ToolchainEvidenceLevel.PROFILING],
            artifact_types=[ToolchainArtifactType.EXECUTABLE_RUN],
            hardware_generations=["RDNA 4", "CDNA 3", "CDNA 4"],
            gpu_arch_patterns=["gfx*"],
            rocm_version_min="6.2",
            expected_binaries=["rocprofv3"],
            probe_command=["rocprofv3", "--version"],
            source_refs=[
                "https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html",
                "https://github.com/ROCm/rocm-systems",
            ],
            notes="Profiling evidence only; not score or correctness authority.",
        ),
        ToolchainCapability(
            tool_id="rocprofv3-avail",
            display_name="ROCprofiler SDK rocprofv3-avail",
            lifecycle=ToolLifecycle.ACTIVE,
            evidence_levels=[ToolchainEvidenceLevel.PROFILING],
            artifact_types=[ToolchainArtifactType.EXECUTABLE_RUN],
            hardware_generations=["RDNA 4", "CDNA 3", "CDNA 4"],
            gpu_arch_patterns=["gfx*"],
            rocm_version_min="6.2",
            expected_binaries=["rocprofv3-avail"],
            probe_command=["rocprofv3-avail", "--help"],
            source_refs=[
                "https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3-avail.html",
                "https://github.com/ROCm/rocm-systems",
            ],
            notes="Counter/configuration discovery companion to rocprofv3.",
        ),
        ToolchainCapability(
            tool_id="rocprofiler-systems",
            display_name="ROCm Systems Profiler legacy repository",
            lifecycle=ToolLifecycle.MIGRATED,
            replacement_tool_id="rocm-systems",
            evidence_levels=[
                ToolchainEvidenceLevel.RUNTIME,
                ToolchainEvidenceLevel.PROFILING,
            ],
            artifact_types=[ToolchainArtifactType.EXECUTABLE_RUN],
            hardware_generations=["RDNA 4", "CDNA 3", "CDNA 4"],
            gpu_arch_patterns=["gfx*"],
            source_refs=[
                "https://github.com/ROCm/rocprofiler-systems",
                "https://github.com/ROCm/rocm-systems",
            ],
            notes="Historical repository; source of truth migrated to ROCm Systems.",
        ),
        ToolchainCapability(
            tool_id="rocm-systems",
            display_name="ROCm Systems super-repo",
            lifecycle=ToolLifecycle.ACTIVE,
            evidence_levels=[
                ToolchainEvidenceLevel.RUNTIME,
                ToolchainEvidenceLevel.PROFILING,
            ],
            artifact_types=[ToolchainArtifactType.EXECUTABLE_RUN],
            hardware_generations=["RDNA 4", "CDNA 3", "CDNA 4"],
            gpu_arch_patterns=["gfx*"],
            source_refs=["https://github.com/ROCm/rocm-systems"],
            notes="Repository source-of-truth signal, not a directly executed tool.",
        ),
        ToolchainCapability(
            tool_id="rocminfo",
            display_name="rocminfo",
            lifecycle=ToolLifecycle.ACTIVE,
            evidence_levels=[ToolchainEvidenceLevel.RUNTIME],
            artifact_types=[ToolchainArtifactType.NONE],
            hardware_generations=["RDNA 4", "CDNA 3", "CDNA 4"],
            gpu_arch_patterns=["gfx*"],
            expected_binaries=["rocminfo"],
            probe_command=["rocminfo"],
            source_refs=["https://github.com/ROCm/rocm-systems"],
            notes="Runtime/device discovery, not compiler evidence.",
        ),
        ToolchainCapability(
            tool_id="rocm_agent_enumerator",
            display_name="rocm_agent_enumerator",
            lifecycle=ToolLifecycle.ACTIVE,
            evidence_levels=[ToolchainEvidenceLevel.RUNTIME],
            artifact_types=[ToolchainArtifactType.NONE],
            hardware_generations=["RDNA 4", "CDNA 3", "CDNA 4"],
            gpu_arch_patterns=["gfx*"],
            expected_binaries=["rocm_agent_enumerator"],
            probe_command=["rocm_agent_enumerator"],
            source_refs=["https://github.com/ROCm/rocm-systems"],
            notes="Architecture discovery helper.",
        ),
        ToolchainCapability(
            tool_id="readelf",
            display_name="readelf",
            lifecycle=ToolLifecycle.ACTIVE,
            evidence_levels=[ToolchainEvidenceLevel.STATIC],
            artifact_types=[
                ToolchainArtifactType.ELF_OBJECT,
                ToolchainArtifactType.ROCM_BINARY,
                ToolchainArtifactType.STATIC_FUTURE,
            ],
            hardware_generations=["RDNA 4", "CDNA 3", "CDNA 4"],
            gpu_arch_patterns=["gfx*"],
            expected_binaries=["readelf"],
            probe_command=["readelf", "--version"],
            source_refs=["https://sourceware.org/binutils/docs/binutils/readelf.html"],
            notes="Planned v1.17 fallback for ELF metadata.",
        ),
        ToolchainCapability(
            tool_id="llvm-objdump",
            display_name="LLVM objdump",
            lifecycle=ToolLifecycle.ACTIVE,
            evidence_levels=[ToolchainEvidenceLevel.STATIC],
            artifact_types=[
                ToolchainArtifactType.ELF_OBJECT,
                ToolchainArtifactType.ROCM_BINARY,
                ToolchainArtifactType.STATIC_FUTURE,
            ],
            hardware_generations=["RDNA 4", "CDNA 3", "CDNA 4"],
            gpu_arch_patterns=["gfx*"],
            expected_binaries=["llvm-objdump"],
            probe_command=["llvm-objdump", "--version"],
            source_refs=["https://llvm.org/docs/CommandGuide/llvm-objdump.html"],
            notes="Planned v1.17 object inspection route.",
        ),
        ToolchainCapability(
            tool_id="roc-objdump",
            display_name="roc-objdump",
            lifecycle=ToolLifecycle.CANDIDATE,
            evidence_levels=[ToolchainEvidenceLevel.STATIC],
            artifact_types=[
                ToolchainArtifactType.ELF_OBJECT,
                ToolchainArtifactType.ROCM_BINARY,
                ToolchainArtifactType.STATIC_FUTURE,
            ],
            hardware_generations=["RDNA 4", "CDNA 3", "CDNA 4"],
            gpu_arch_patterns=["gfx*"],
            expected_binaries=["roc-objdump"],
            probe_command=["roc-objdump", "--version"],
            source_refs=[
                "https://rocm.docs.amd.com/projects/HIP/en/develop/understand/compilers.html"
            ],
            notes="Distribution-dependent candidate for v1.17 static evidence.",
        ),
        ToolchainCapability(
            tool_id="rga",
            display_name="Radeon GPU Analyzer",
            lifecycle=ToolLifecycle.PLANNED,
            evidence_levels=[ToolchainEvidenceLevel.STATIC],
            artifact_types=[
                ToolchainArtifactType.HIP_COMPILER_OUTPUT,
                ToolchainArtifactType.ROCM_BINARY,
                ToolchainArtifactType.STATIC_FUTURE,
            ],
            hardware_generations=["RDNA 4", "CDNA 3", "CDNA 4"],
            gpu_arch_patterns=["gfx*"],
            expected_binaries=["rga"],
            probe_command=["rga", "--version"],
            source_refs=[
                "https://github.com/GPUOpen-Tools/radeon_gpu_analyzer",
                "https://gpuopen.com/manuals/rga_manual/help_manual/",
            ],
            notes="Planned v1.17 compiler-facing static evidence route.",
        ),
    ]


def build_toolchain_routing_report(
    request: ToolchainRoutingRequest,
    *,
    registry: list[ToolchainCapability] | None = None,
    runner: ProbeRunner | None = None,
    which: Which = shutil.which,
    timeout_seconds: float = DEFAULT_TOOLCHAIN_PROBE_TIMEOUT_SECONDS,
    now: Callable[[], datetime] | None = None,
) -> ToolchainRoutingReport:
    """Build a diagnostic routing report for a requested evidence path."""

    generated_at = (now or (lambda: datetime.now(UTC)))().isoformat()
    decisions: list[ToolchainRoutingDecision] = []
    effective_registry = registry or default_toolchain_registry()
    for capability in effective_registry:
        decision = _decision_for_capability(
            request,
            capability,
            runner=runner,
            which=which,
            timeout_seconds=timeout_seconds,
        )
        if decision is not None:
            decisions.append(decision)

    selected_tool_id = None
    for index, decision in enumerate(decisions):
        if decision.status == ToolchainStatus.AVAILABLE:
            selected_tool_id = decision.tool_id
            decisions[index] = decision.model_copy(update={"selected": True})
            break

    if selected_tool_id is None and not decisions:
        decisions.append(
            ToolchainRoutingDecision(
                tool_id="none",
                lifecycle=ToolLifecycle.REJECTED,
                status=ToolchainStatus.UNSUPPORTED_ARTIFACT,
                reason_code="no_registry_entry",
                reason=(
                    "No registry entry supports the requested evidence level "
                    f"{request.evidence_level.value} and artifact "
                    f"{request.artifact_type.value}."
                ),
            )
        )

    return ToolchainRoutingReport(
        generated_at=generated_at,
        request=request,
        selected_tool_id=selected_tool_id,
        decisions=decisions,
    )


def _decision_for_capability(
    request: ToolchainRoutingRequest,
    capability: ToolchainCapability,
    *,
    runner: ProbeRunner | None,
    which: Which,
    timeout_seconds: float,
) -> ToolchainRoutingDecision | None:
    if request.evidence_level not in capability.evidence_levels:
        return None
    if request.artifact_type not in capability.artifact_types:
        return ToolchainRoutingDecision(
            tool_id=capability.tool_id,
            lifecycle=capability.lifecycle,
            status=ToolchainStatus.UNSUPPORTED_ARTIFACT,
            reason_code="unsupported_artifact",
            reason=(
                f"{capability.tool_id} does not support artifact type "
                f"{request.artifact_type.value} for {request.evidence_level.value}."
            ),
            source_refs=capability.source_refs,
        )
    if request.gpu_architecture and not _supports_arch(
        capability, request.gpu_architecture
    ):
        return ToolchainRoutingDecision(
            tool_id=capability.tool_id,
            lifecycle=capability.lifecycle,
            status=ToolchainStatus.UNSUPPORTED_ARCH,
            reason_code="unsupported_arch",
            reason=(
                f"{capability.tool_id} does not declare support for "
                f"{request.gpu_architecture}."
            ),
            source_refs=capability.source_refs,
        )
    if capability.lifecycle == ToolLifecycle.DEPRECATED:
        return _lifecycle_decision(capability, ToolchainStatus.DEPRECATED)
    if capability.lifecycle == ToolLifecycle.MIGRATED:
        return _lifecycle_decision(capability, ToolchainStatus.MIGRATED)
    if capability.lifecycle == ToolLifecycle.PLANNED:
        return _lifecycle_decision(capability, ToolchainStatus.PLANNED)
    if capability.lifecycle == ToolLifecycle.REJECTED:
        return _lifecycle_decision(capability, ToolchainStatus.REJECTED)
    if not capability.expected_binaries:
        return ToolchainRoutingDecision(
            tool_id=capability.tool_id,
            lifecycle=capability.lifecycle,
            status=ToolchainStatus.UNAVAILABLE,
            reason_code="repository_reference_only",
            reason=f"{capability.tool_id} is a source reference, not a runnable tool.",
            source_refs=capability.source_refs,
        )

    binary = capability.expected_binaries[0]
    command = capability.probe_command or [binary, "--version"]
    probe = probe_toolchain_tool(
        capability.tool_id,
        binary,
        command,
        runner=runner,
        which=which,
        timeout_seconds=timeout_seconds,
    )
    return ToolchainRoutingDecision(
        tool_id=capability.tool_id,
        lifecycle=capability.lifecycle,
        status=probe.status,
        reason_code=f"probe_{probe.status.value}",
        reason=(
            f"{capability.tool_id} probe {probe.status.value}."
            if probe.status != ToolchainStatus.UNAVAILABLE
            else f"{binary} is not available on PATH."
        ),
        source_refs=capability.source_refs,
        probe=probe,
    )


def probe_toolchain_tool(
    tool_id: str,
    binary: str,
    command: list[str],
    *,
    runner: ProbeRunner | None = None,
    which: Which = shutil.which,
    timeout_seconds: float = DEFAULT_TOOLCHAIN_PROBE_TIMEOUT_SECONDS,
) -> ToolchainProbeResult:
    """Run one bounded toolchain probe."""

    path = which(binary)
    if path is None:
        return ToolchainProbeResult(
            tool_id=tool_id,
            command=command,
            status=ToolchainStatus.UNAVAILABLE,
            timeout_seconds=timeout_seconds,
        )
    effective_runner = runner or _run_probe
    try:
        completed = effective_runner(command, timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        return ToolchainProbeResult(
            tool_id=tool_id,
            command=command,
            path=path,
            status=ToolchainStatus.FAILED,
            stdout_tail=_tail(exc.stdout),
            stderr_tail=_tail(exc.stderr),
            timeout_seconds=timeout_seconds,
        )
    except OSError as exc:
        return ToolchainProbeResult(
            tool_id=tool_id,
            command=command,
            path=path,
            status=ToolchainStatus.FAILED,
            stderr_tail=_tail(str(exc)),
            timeout_seconds=timeout_seconds,
        )

    return ToolchainProbeResult(
        tool_id=tool_id,
        command=command,
        path=path,
        status=(
            ToolchainStatus.AVAILABLE
            if completed.returncode == 0
            else ToolchainStatus.FAILED
        ),
        returncode=completed.returncode,
        stdout_tail=_tail(completed.stdout),
        stderr_tail=_tail(completed.stderr),
        timeout_seconds=timeout_seconds,
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


def _supports_arch(capability: ToolchainCapability, gpu_architecture: str) -> bool:
    if not capability.gpu_arch_patterns:
        return True
    return any(
        fnmatch.fnmatchcase(gpu_architecture, pattern)
        for pattern in capability.gpu_arch_patterns
    )


def _lifecycle_decision(
    capability: ToolchainCapability,
    status: ToolchainStatus,
) -> ToolchainRoutingDecision:
    reason_code = f"tool_{status.value}"
    replacement = (
        f" Use {capability.replacement_tool_id} instead."
        if capability.replacement_tool_id
        else ""
    )
    return ToolchainRoutingDecision(
        tool_id=capability.tool_id,
        lifecycle=capability.lifecycle,
        status=status,
        reason_code=reason_code,
        reason=f"{capability.tool_id} lifecycle is {status.value}.{replacement}",
        fallback_tool_id=capability.replacement_tool_id,
        source_refs=capability.source_refs,
    )


def _tail(value: object, limit: int = 4000) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        text = value.decode(errors="replace")
    else:
        text = str(value)
    return text[-limit:]

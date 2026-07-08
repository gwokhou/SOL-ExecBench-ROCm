# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""ROCm toolchain routing models."""

from __future__ import annotations

from collections.abc import Callable
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

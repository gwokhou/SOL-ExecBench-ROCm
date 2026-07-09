# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Strict static kernel evidence sidecar models."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import BeforeValidator, ConfigDict, Field

from sol_execbench.core.data.base_model import BaseModelWithDocstrings


STATIC_KERNEL_EVIDENCE_SCHEMA_VERSION = "sol_execbench.static_kernel_evidence.v2"
_STATIC_MODEL_CONFIG = ConfigDict(
    extra="forbid",
    frozen=True,
    strict=True,
    use_attribute_docstrings=True,
)


class StaticKernelEvidenceStatus(str, Enum):
    """Aggregate and per-artifact status vocabulary."""

    COLLECTED = "collected"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    SKIPPED = "skipped"


class StaticKernelEvidenceReasonCode(str, Enum):
    """Stable reason-code vocabulary for static evidence outcomes."""

    STATIC_EVIDENCE_NOT_REQUESTED = "static_evidence_not_requested"
    STATIC_EVIDENCE_COLLECTED = "static_evidence_collected"
    PARTIAL_ARTIFACT_METADATA = "partial_artifact_metadata"
    PARTIAL_DISASSEMBLY_ONLY = "partial_disassembly_only"
    PARTIAL_METADATA_ONLY = "partial_metadata_only"
    ARTIFACT_UNAVAILABLE = "artifact_unavailable"
    TOOLCHAIN_UNAVAILABLE = "toolchain_unavailable"
    UNSUPPORTED_SOLUTION_TYPE = "unsupported_solution_type"
    UNSUPPORTED_ARCHITECTURE = "unsupported_architecture"
    UNSUPPORTED_ARTIFACT_TYPE = "unsupported_artifact_type"
    EXTRACTOR_FAILED = "extractor_failed"
    EXTRACTOR_TIMEOUT = "extractor_timeout"
    PARSER_FAILED = "parser_failed"


def _validate_status(value: object) -> object:
    if isinstance(value, str):
        return StaticKernelEvidenceStatus(value)
    return value


def _validate_reason_code(value: object) -> object:
    if isinstance(value, str):
        return StaticKernelEvidenceReasonCode(value)
    return value


StaticKernelEvidenceStatusField = Annotated[
    StaticKernelEvidenceStatus,
    BeforeValidator(_validate_status),
]
StaticKernelEvidenceReasonCodeField = Annotated[
    StaticKernelEvidenceReasonCode,
    BeforeValidator(_validate_reason_code),
]


class StaticKernelEvidenceSourceReference(BaseModelWithDocstrings):
    """Source reference used to explain static evidence metadata."""

    model_config = _STATIC_MODEL_CONFIG

    kind: str
    """Reference kind, such as artifact, tool, parser, or note."""
    value: str
    """Reference value or identifier."""
    description: str = ""
    """Human-readable reference description."""


class StaticKernelEvidenceWarning(BaseModelWithDocstrings):
    """Nonfatal warning attached to a static evidence sidecar."""

    model_config = _STATIC_MODEL_CONFIG

    code: str
    """Stable warning code."""
    message: str
    """Human-readable warning message."""
    source_reference: StaticKernelEvidenceSourceReference | None = None
    """Optional reference that explains the warning."""


class StaticKernelEvidenceClassification(BaseModelWithDocstrings):
    """Conservative artifact classification metadata."""

    model_config = _STATIC_MODEL_CONFIG

    metadata_present: bool = False
    """Whether artifact metadata was present."""
    disassembly_present: bool = False
    """Whether disassembly text was present."""
    detected_architectures: list[str] = Field(default_factory=list)
    """Detected gfx architecture identifiers."""
    symbol_count: int | None = Field(default=None, ge=0)
    """Number of detected symbols when known."""


class StaticResourceFootprintIdentity(BaseModelWithDocstrings):
    """Freshness identity for a static resource footprint."""

    model_config = _STATIC_MODEL_CONFIG

    artifact_id: str
    """Artifact this footprint was derived from."""
    extractor_tool_id: str
    """Extractor tool that produced the footprint."""
    source_sha256: str | None = None
    """SHA256 of the source artifact content when available."""
    generated_at: str | None = None
    """UTC timestamp when the footprint was generated, when available."""


class StaticResourceFootprint(BaseModelWithDocstrings):
    """Per-kernel resource footprint derived from routed static extractors."""

    model_config = _STATIC_MODEL_CONFIG

    identity: StaticResourceFootprintIdentity | None = None
    """Freshness identity for this footprint when available."""
    vgpr_used: int | None = Field(default=None, ge=0)
    """Vector general-purpose registers used by the kernel when known."""
    sgpr_used: int | None = Field(default=None, ge=0)
    """Scalar general-purpose registers used by the kernel when known."""
    lds_bytes: int | None = Field(default=None, ge=0)
    """Local Data Share bytes allocated by the kernel when known."""
    scratch_bytes: int | None = Field(default=None, ge=0)
    """Scratch memory bytes spilled by the kernel when known."""
    spill_detected: bool | None = None
    """Whether register spilling was detected when known."""
    occupancy_estimate_waves_per_cu: int | None = Field(default=None, ge=0)
    """Conservative occupancy estimate in waves per CU when known."""
    wavefront_size: int | None = Field(default=None, ge=0)
    """Wavefront size assumed by the footprint when known."""
    source_tool: str | None = None
    """Tool that produced this footprint, such as ``roc-objdump``."""
    source_confidence: str | None = None
    """Confidence of the footprint values when known."""
    diagnostic_only: Literal[True] = True
    """Resource footprint is diagnostic metadata only."""
    correctness_authority: Literal[False] = False
    """Resource footprint never proves correctness."""
    performance_authority: Literal[False] = False
    """Resource footprint never proves performance."""
    timing_authority: Literal[False] = False
    """Resource footprint never proves benchmark timing."""
    score_authority: Literal[False] = False
    """Resource footprint never proves score validity."""
    paper_parity_authority: Literal[False] = False
    """Resource footprint never proves paper parity."""
    leaderboard_authority: Literal[False] = False
    """Resource footprint never proves leaderboard readiness."""
    source_references: list[StaticKernelEvidenceSourceReference] = Field(
        default_factory=list
    )
    """References supporting this footprint."""


class StaticKernelEvidenceKernel(BaseModelWithDocstrings):
    """Conservative kernel symbol metadata."""

    model_config = _STATIC_MODEL_CONFIG

    name: str
    """Kernel or symbol name."""
    demangled_name: str | None = None
    """Demangled symbol name when known."""
    detected_architectures: list[str] = Field(default_factory=list)
    """Architectures associated with this kernel."""
    source_references: list[StaticKernelEvidenceSourceReference] = Field(
        default_factory=list
    )
    """References supporting this kernel metadata."""
    footprint: StaticResourceFootprint | None = None
    """Per-kernel resource footprint when routed extractors produced one."""


class StaticKernelEvidenceArtifact(BaseModelWithDocstrings):
    """One static evidence artifact entry."""

    model_config = _STATIC_MODEL_CONFIG

    artifact_id: str
    """Stable artifact identifier within the sidecar."""
    artifact_type: str
    """Artifact class such as elf_object, rocm_binary, or disassembly."""
    status: StaticKernelEvidenceStatusField
    """Per-artifact status."""
    reason_code: StaticKernelEvidenceReasonCodeField | None = None
    """Per-artifact reason code."""
    source_path: str | None = None
    """Source artifact path recorded by a future extractor."""
    persisted_path: str | None = None
    """Persisted sidecar-relative artifact path recorded by a future extractor."""
    size_bytes: int | None = Field(default=None, ge=0)
    """Artifact size when known."""
    sha256: str | None = None
    """Artifact digest when known."""
    producer: str | None = None
    """Producer that created or registered this artifact when known."""
    target_architecture: str | None = None
    """Target gfx architecture when known."""
    inspectable: bool = False
    """Whether later static tools may inspect this artifact."""
    classification: StaticKernelEvidenceClassification = Field(
        default_factory=StaticKernelEvidenceClassification
    )
    """Conservative artifact classification."""
    source_references: list[StaticKernelEvidenceSourceReference] = Field(
        default_factory=list
    )
    """References supporting this artifact entry."""


class StaticKernelEvidenceToolRun(BaseModelWithDocstrings):
    """One static evidence tool-run record."""

    model_config = _STATIC_MODEL_CONFIG

    tool_id: str
    """Stable tool identifier."""
    command: list[str] = Field(default_factory=list)
    """Command that a future extractor records."""
    status: StaticKernelEvidenceStatusField
    """Tool-run status."""
    reason_code: StaticKernelEvidenceReasonCodeField | None = None
    """Tool-run reason code."""
    returncode: int | None = None
    """Process return code when a future extractor records one."""
    stdout_tail: str = ""
    """Bounded stdout tail recorded by a future extractor."""
    stderr_tail: str = ""
    """Bounded stderr tail recorded by a future extractor."""
    timeout_seconds: float | None = Field(default=None, ge=0.0)
    """Timeout used by a future extractor when known."""
    raw_output_path: str | None = None
    """Sidecar-relative bounded raw output artifact path when preserved."""


class StaticKernelEvidenceSidecar(BaseModelWithDocstrings):
    """Strict diagnostic-only static kernel evidence sidecar."""

    model_config = _STATIC_MODEL_CONFIG

    schema_version: Literal["sol_execbench.static_kernel_evidence.v2"] = (
        STATIC_KERNEL_EVIDENCE_SCHEMA_VERSION
    )
    """Static kernel evidence schema version."""
    status: StaticKernelEvidenceStatusField
    """Aggregate static evidence status."""
    reason_code: StaticKernelEvidenceReasonCodeField
    """Aggregate reason code."""
    diagnostic_only: Literal[True] = True
    """Static evidence is diagnostic metadata only."""
    correctness_authority: Literal[False] = False
    """Static evidence never proves correctness."""
    performance_authority: Literal[False] = False
    """Static evidence never proves performance."""
    timing_authority: Literal[False] = False
    """Static evidence never proves benchmark timing."""
    score_authority: Literal[False] = False
    """Static evidence never proves score validity."""
    paper_parity_authority: Literal[False] = False
    """Static evidence never proves paper parity."""
    leaderboard_authority: Literal[False] = False
    """Static evidence never proves leaderboard readiness."""
    classification: StaticKernelEvidenceClassification = Field(
        default_factory=StaticKernelEvidenceClassification
    )
    """Aggregate conservative classification."""
    artifacts: list[StaticKernelEvidenceArtifact] = Field(default_factory=list)
    """Static evidence artifact entries."""
    tool_runs: list[StaticKernelEvidenceToolRun] = Field(default_factory=list)
    """Static evidence tool-run records."""
    kernels: list[StaticKernelEvidenceKernel] = Field(default_factory=list)
    """Conservative kernel metadata entries."""
    footprints: list[StaticResourceFootprint] = Field(default_factory=list)
    """Per-kernel resource footprints when routed extractors produced them."""
    warnings: list[StaticKernelEvidenceWarning] = Field(default_factory=list)
    """Nonfatal static evidence warnings."""
    source_references: list[StaticKernelEvidenceSourceReference] = Field(
        default_factory=list
    )
    """References supporting this sidecar."""

    def to_dict(self) -> dict[str, object]:
        """Return the JSON-compatible sidecar payload."""
        return self.model_dump(mode="json")

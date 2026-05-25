# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict diagnostic-only static kernel evidence sidecar contract."""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from typing import Annotated, Literal

from pydantic import BeforeValidator, ConfigDict, Field

from sol_execbench.core.data.base_model import BaseModelWithDocstrings


STATIC_KERNEL_EVIDENCE_SCHEMA_VERSION = "sol_execbench.static_kernel_evidence.v1"
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


class StaticKernelEvidenceSidecar(BaseModelWithDocstrings):
    """Strict diagnostic-only static kernel evidence sidecar."""

    model_config = _STATIC_MODEL_CONFIG

    schema_version: Literal[STATIC_KERNEL_EVIDENCE_SCHEMA_VERSION] = (
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
    warnings: list[StaticKernelEvidenceWarning] = Field(default_factory=list)
    """Nonfatal static evidence warnings."""
    source_references: list[StaticKernelEvidenceSourceReference] = Field(
        default_factory=list
    )
    """References supporting this sidecar."""

    def to_dict(self) -> dict[str, object]:
        """Return the JSON-compatible sidecar payload."""
        return self.model_dump(mode="json")


def build_static_kernel_evidence_sidecar(
    *,
    status: StaticKernelEvidenceStatus | str,
    reason_code: StaticKernelEvidenceReasonCode | str,
    classification: StaticKernelEvidenceClassification | None = None,
    artifacts: Sequence[StaticKernelEvidenceArtifact] = (),
    tool_runs: Sequence[StaticKernelEvidenceToolRun] = (),
    kernels: Sequence[StaticKernelEvidenceKernel] = (),
    warnings: Sequence[StaticKernelEvidenceWarning] = (),
    source_references: Sequence[StaticKernelEvidenceSourceReference] = (),
) -> StaticKernelEvidenceSidecar:
    """Build a strict static evidence sidecar without collecting artifacts."""

    return StaticKernelEvidenceSidecar(
        status=status,
        reason_code=reason_code,
        classification=classification or StaticKernelEvidenceClassification(),
        artifacts=list(artifacts),
        tool_runs=list(tool_runs),
        kernels=list(kernels),
        warnings=list(warnings),
        source_references=list(source_references),
    )


def build_static_kernel_evidence_skipped(
    reason_code: StaticKernelEvidenceReasonCode | str = (
        StaticKernelEvidenceReasonCode.STATIC_EVIDENCE_NOT_REQUESTED
    ),
) -> StaticKernelEvidenceSidecar:
    """Build a skipped sidecar for unrequested static evidence."""

    return build_static_kernel_evidence_sidecar(
        status=StaticKernelEvidenceStatus.SKIPPED,
        reason_code=reason_code,
    )


def build_static_kernel_evidence_unavailable(
    reason_code: StaticKernelEvidenceReasonCode | str = (
        StaticKernelEvidenceReasonCode.TOOLCHAIN_UNAVAILABLE
    ),
) -> StaticKernelEvidenceSidecar:
    """Build an unavailable sidecar for missing optional tooling or artifacts."""

    return build_static_kernel_evidence_sidecar(
        status=StaticKernelEvidenceStatus.UNAVAILABLE,
        reason_code=reason_code,
    )


def build_static_kernel_evidence_unsupported(
    reason_code: StaticKernelEvidenceReasonCode | str = (
        StaticKernelEvidenceReasonCode.UNSUPPORTED_SOLUTION_TYPE
    ),
) -> StaticKernelEvidenceSidecar:
    """Build an unsupported sidecar for unsupported solution or artifact classes."""

    return build_static_kernel_evidence_sidecar(
        status=StaticKernelEvidenceStatus.UNSUPPORTED,
        reason_code=reason_code,
    )


def build_static_kernel_evidence_failed(
    reason_code: StaticKernelEvidenceReasonCode | str = (
        StaticKernelEvidenceReasonCode.EXTRACTOR_FAILED
    ),
) -> StaticKernelEvidenceSidecar:
    """Build a failed sidecar for failed optional evidence extraction."""

    return build_static_kernel_evidence_sidecar(
        status=StaticKernelEvidenceStatus.FAILED,
        reason_code=reason_code,
    )


def build_static_kernel_evidence_partial(
    reason_code: StaticKernelEvidenceReasonCode | str = (
        StaticKernelEvidenceReasonCode.PARTIAL_ARTIFACT_METADATA
    ),
) -> StaticKernelEvidenceSidecar:
    """Build a partial sidecar for incomplete optional static evidence."""

    return build_static_kernel_evidence_sidecar(
        status=StaticKernelEvidenceStatus.PARTIAL,
        reason_code=reason_code,
    )

# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Pydantic models and enums for decision sidecars.

The Decision sidecar turns decision-ready data-layer facts (an ``ArchIsaBudget``
plus per-kernel ``StaticResourceFootprint`` records) into confidence-weighted
Layer R optimization hints. It is diagnostic-only and never re-asserts benchmark
authority. See ``docs/decision_sidecar_contract.md`` and
``docs/decision-modeling-research.md`` for the contract and modeling survey.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import ConfigDict, Field

from sol_execbench.core.bench.diagnostic_sidecar import DiagnosticSidecarAuthority
from sol_execbench.core.data.base_model import BaseModelWithDocstrings


DECISION_SCHEMA_VERSION = "sol_execbench.decision.v1"
_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)


class DecisionStatus(str, Enum):
    """Aggregate decision sidecar availability."""

    AVAILABLE = "available"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"


class DecisionReasonCode(str, Enum):
    """Stable reason-code vocabulary for decision rendering."""

    DECISION_RENDERED = "decision_rendered"
    PARTIAL_DECISION = "partial_decision"
    NO_DECISION_INPUTS = "no_decision_inputs"
    DECISION_UNAVAILABLE = "decision_unavailable"


class DecisionBottleneckClass(str, Enum):
    """Closed Layer R (static-inferable resource) bottleneck vocabulary.

    Static derivation emits Layer R only. Compute-bound / memory-bound /
    latency-bound verdicts require runtime profiling and are never produced
    from static facts (see decision-modeling-research.md §5, §8.1).
    """

    REGISTER_PRESSURE_HIGH = "register_pressure_high"
    LDS_PRESSURE_HIGH = "lds_pressure_high"
    SPILL_DETECTED = "spill_detected"
    WORKGROUP_SIZE_LIMITED = "workgroup_size_limited"
    BARRIER_LIMITED = "barrier_limited"
    WAVEFRONT_MISALIGNED = "wavefront_misaligned"
    CACHE_LINE_MISALIGNED = "cache_line_misaligned"


class DecisionConfidence(str, Enum):
    """Confidence vocabulary for inferred decision hints.

    The ``inferred_*`` prefix marks these as static-inferred (not measured).
    Runtime ``measured_*`` values live in ``profile_summary.v2`` and take
    precedence on conflict (see decision_sidecar_contract.md precedence rule).
    """

    INFERRED_HIGH = "inferred_high"
    INFERRED_MEDIUM = "inferred_medium"
    INFERRED_LOW = "inferred_low"


class DecisionFreshnessStatus(str, Enum):
    """Freshness validation status for a decision sidecar."""

    CURRENT = "current"
    STALE = "stale"
    UNKNOWN = "unknown"


class DecisionGovernanceStatus(str, Enum):
    """Diagnostic governance status for a decision sidecar."""

    USABLE_DIAGNOSTIC = "usable_diagnostic"
    STALE_DIAGNOSTIC = "stale_diagnostic"
    UNAVAILABLE = "unavailable"
    INVALID_DIAGNOSTIC = "invalid_diagnostic"


class DecisionSourceRef(BaseModelWithDocstrings):
    """Compact reference to source evidence used by the decision sidecar."""

    model_config = _MODEL_CONFIG

    kind: str
    """Evidence kind such as static_evidence, environment, or profile."""
    label: str
    """Stable compact label for the evidence source."""
    status: str | None = None
    """Optional source status."""


class DecisionArtifactCitation(BaseModelWithDocstrings):
    """Compact sidecar artifact citation."""

    model_config = _MODEL_CONFIG

    kind: str
    """Artifact kind such as static_evidence or environment."""
    label: str
    """Compact artifact label."""
    path: str | None = None
    """Compact path, normally a file name or relative path."""
    sha256: str | None = None
    """Artifact checksum when available."""
    status: str | None = None
    """Optional source status."""


class DecisionIdentity(BaseModelWithDocstrings):
    """Freshness identity for a generated decision sidecar."""

    model_config = _MODEL_CONFIG

    generated_at: str
    """UTC timestamp when the sidecar was generated."""
    sol_version: str
    """Producer/runtime SOL version or HIP-facing supported SOL tag."""
    trace_path: str | None = None
    """Compact trace path or file name when available."""
    target_id: str | None = None
    """Optional target/run denominator identity."""
    run_id: str | None = None
    """Optional run identity."""
    candidate_id: str | None = None
    """Canonical candidate identity."""
    source_sha256: str | None = None
    """Canonical source-content SHA256 identity."""


class DecisionFreshnessValidation(BaseModelWithDocstrings):
    """Result of validating decision freshness identity."""

    model_config = _MODEL_CONFIG

    status: DecisionFreshnessStatus
    """Freshness result."""
    reason_codes: list[str] = Field(default_factory=list)
    """Stable reason codes explaining stale or unknown status."""


class DecisionGovernanceGuardrail(DiagnosticSidecarAuthority):
    """Authority boundary after optional decision governance checks."""

    status: DecisionGovernanceStatus
    """Diagnostic governance status."""
    reason_codes: list[str] = Field(default_factory=list)
    """Stable reason codes for unavailable, stale, or invalid states."""


class DecisionHintIdentity(BaseModelWithDocstrings):
    """Per-hint provenance linking a hint back to its source footprint."""

    model_config = _MODEL_CONFIG

    artifact_id: str
    """Artifact the hint was derived from, such as a static-evidence kernel id."""
    extractor_tool_id: str | None = None
    """Tool that produced the underlying footprint, such as roc-objdump."""
    source_sha256: str | None = None
    """SHA256 of the source artifact content when available."""
    generated_at: str | None = None
    """UTC timestamp when the hint was generated, when available."""


class DecisionHint(BaseModelWithDocstrings):
    """One confidence-weighted Layer R optimization hint.

    Diagnostic only; never asserts correctness, performance, or benchmark
    authority. Static hints are resource risk signals (most actionable for
    latency-bound kernels) and must be confirmed via runtime profiling before
    acting on occupancy-raising recommendations (decision-modeling-research.md §7).
    """

    model_config = _MODEL_CONFIG

    bottleneck_class: DecisionBottleneckClass
    """Closed Layer R bottleneck label."""
    confidence: DecisionConfidence
    """Inferred confidence; ``inferred_*`` marks static (not measured)."""
    message: str
    """Bounded diagnostic message describing the observed signal."""
    recommendation: str | None = None
    """Prompt-safe next-experiment recommendation sourced from AMD HIP guidelines."""
    architecture: str | None = None
    """gfx architecture the hint was derived for, when known."""
    identity: DecisionHintIdentity | None = None
    """Per-hint provenance linking back to the source footprint."""
    limitations: list[str] = Field(default_factory=list)
    """Per-hint limitations, such as the dynamic-allocation static-derivation gap."""
    evidence_refs: list[str] = Field(default_factory=list)
    """Upstream evidence references supporting the hint."""
    source_refs: list[DecisionSourceRef] = Field(default_factory=list)
    """Compact source references supporting this hint."""


class DecisionSummary(BaseModelWithDocstrings):
    """Compact aggregate decision summary."""

    model_config = _MODEL_CONFIG

    hint_count: int = Field(ge=0)
    """Number of Layer R hints rendered."""
    footprint_count: int = Field(ge=0)
    """Number of static footprints considered."""
    architecture: str | None = None
    """gfx architecture budget used, when known."""
    bottleneck_counts: dict[str, int] = Field(default_factory=dict)
    """Counts per bottleneck class label."""


class DecisionSidecar(BaseModelWithDocstrings):
    """Strict diagnostic-only sidecar for Layer R optimization hints."""

    model_config = _MODEL_CONFIG

    schema_version: Literal["sol_execbench.decision.v1"] = DECISION_SCHEMA_VERSION
    status: DecisionStatus
    reason_code: DecisionReasonCode
    identity: DecisionIdentity
    authority: Literal["diagnostic"] = "diagnostic"
    summary: DecisionSummary
    hints: list[DecisionHint] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    source_refs: list[DecisionSourceRef] = Field(default_factory=list)
    artifact_citations: list[DecisionArtifactCitation] = Field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Return the JSON-compatible sidecar payload."""
        return self.model_dump(mode="json")

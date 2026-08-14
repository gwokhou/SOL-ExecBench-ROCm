# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Current performance evidence and optimization artifact schemas."""

from enum import StrEnum


class PerformanceArtifactSchema(StrEnum):
    """Canonical performance evidence and optimization identifiers."""

    AGENT_FEEDBACK = "sol_execbench.agent_feedback.v7"
    DECISION = "sol_execbench.decision.v2"
    PERFORMANCE_DIAGNOSTIC = "sol_execbench.performance_diagnostic.v9"
    PERFORMANCE_EVIDENCE_COMPONENT = (
        "sol_execbench.performance_evidence_component.v1"
    )
    PERFORMANCE_EVIDENCE_MANIFEST = (
        "sol_execbench.performance_evidence_manifest.v5"
    )
    PROFILE_SUMMARY = "sol_execbench.profile_summary.v3"


class PerformanceDiagnosticArtifactKind(StrEnum):
    """Trace-present and no-trace diagnostic variants."""

    NO_TRACE = "no_trace"
    TRACE = "trace"


class PerformanceEvidenceComponentKind(StrEnum):
    """Component artifacts bound by a performance evidence manifest."""

    ACCESS_PATTERN = "access_pattern"
    REPLAY = "replay"
    STATIC_ARTIFACT_MANIFEST = "static_artifact_manifest"
    STATIC_KERNEL = "static_kernel"
    TIMING = "timing"


__all__ = [
    "PerformanceArtifactSchema",
    "PerformanceDiagnosticArtifactKind",
    "PerformanceEvidenceComponentKind",
]

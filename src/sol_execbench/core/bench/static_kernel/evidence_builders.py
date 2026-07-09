# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Static kernel evidence sidecar builders."""

from __future__ import annotations

from collections.abc import Sequence

from sol_execbench.core.bench.static_kernel.evidence_models import (
    StaticKernelEvidenceArtifact,
    StaticKernelEvidenceClassification,
    StaticKernelEvidenceKernel,
    StaticKernelEvidenceReasonCode,
    StaticKernelEvidenceSidecar,
    StaticKernelEvidenceSourceReference,
    StaticKernelEvidenceStatus,
    StaticKernelEvidenceToolRun,
    StaticKernelEvidenceWarning,
)


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
        status=StaticKernelEvidenceStatus(status),
        reason_code=StaticKernelEvidenceReasonCode(reason_code),
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

# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Strict diagnostic-only static kernel evidence sidecar contract."""

from __future__ import annotations

from sol_execbench.core.bench.static_kernel.artifacts import (
    collect_static_kernel_artifacts,
)
from sol_execbench.core.bench.static_kernel.evidence_builders import (
    build_static_kernel_evidence_failed,
    build_static_kernel_evidence_partial,
    build_static_kernel_evidence_sidecar,
    build_static_kernel_evidence_skipped,
    build_static_kernel_evidence_unavailable,
    build_static_kernel_evidence_unsupported,
)
from sol_execbench.core.bench.static_kernel.evidence_models import (
    STATIC_KERNEL_EVIDENCE_SCHEMA_VERSION,
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
from sol_execbench.core.bench.static_kernel.extractors import (
    run_static_kernel_extractors,
)

__all__ = [
    "STATIC_KERNEL_EVIDENCE_SCHEMA_VERSION",
    "StaticKernelEvidenceArtifact",
    "StaticKernelEvidenceClassification",
    "StaticKernelEvidenceKernel",
    "StaticKernelEvidenceReasonCode",
    "StaticKernelEvidenceSidecar",
    "StaticKernelEvidenceSourceReference",
    "StaticKernelEvidenceStatus",
    "StaticKernelEvidenceToolRun",
    "StaticKernelEvidenceWarning",
    "build_static_kernel_evidence_failed",
    "build_static_kernel_evidence_partial",
    "build_static_kernel_evidence_sidecar",
    "build_static_kernel_evidence_skipped",
    "build_static_kernel_evidence_unavailable",
    "build_static_kernel_evidence_unsupported",
    "collect_static_kernel_artifacts",
    "run_static_kernel_extractors",
]

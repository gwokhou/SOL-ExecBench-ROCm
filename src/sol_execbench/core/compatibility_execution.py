# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Compatibility matrix builders and execution classification."""

from __future__ import annotations

from collections.abc import Sequence

from sol_execbench.core.compatibility_entry_models import (
    MatrixClaimBoundary,
    MatrixEntry,
    MatrixExecutionDecision,
)
from sol_execbench.core.compatibility_enums import (
    MatrixCompatibilityReasonCode,
    MatrixCompatibilityStatus,
)
from sol_execbench.core.compatibility_evidence_models import (
    MatrixArtifactReference,
    MatrixObservedEvidence,
    MatrixTarget,
)


def build_matrix_entry(
    *,
    target: MatrixTarget,
    observed: MatrixObservedEvidence,
    status: MatrixCompatibilityStatus | str,
    reason_code: MatrixCompatibilityReasonCode | str,
    reason: str,
    claim_boundary: MatrixClaimBoundary,
    artifacts: Sequence[MatrixArtifactReference] = (),
) -> MatrixEntry:
    """Build a strict Matrix Entry from explicit Target and evidence inputs."""

    return MatrixEntry(
        target=target,
        observed=observed,
        status=MatrixCompatibilityStatus(status),
        reason_code=MatrixCompatibilityReasonCode(reason_code),
        reason=reason,
        claim_boundary=claim_boundary,
        artifacts=list(artifacts),
    )


def classify_matrix_entry_for_execution(
    entry: MatrixEntry,
    *,
    allow_mixed_version_debug: bool = False,
) -> MatrixExecutionDecision:
    """Classify a Matrix Entry into deterministic pre-benchmark allowances."""

    status = entry.status
    reason_code = entry.reason_code
    claims = entry.claim_boundary

    if status is MatrixCompatibilityStatus.MIXED_VERSION:
        if allow_mixed_version_debug:
            return MatrixExecutionDecision(
                status=status,
                reason_code=reason_code,
                benchmark_allowed=False,
                probes_allowed=True,
                smoke_allowed=True,
                reason=(
                    "Mixed-version Target is running under explicit debug override; "
                    "diagnostic probes or smoke execution may continue, but clean "
                    "validation and benchmark claims remain blocked."
                ),
                container_user_space_validated=False,
                native_host_validated=False,
            )
        return MatrixExecutionDecision(
            status=status,
            reason_code=reason_code,
            benchmark_allowed=False,
            probes_allowed=False,
            smoke_allowed=False,
            reason=(
                "Mixed-version Target is blocked before benchmark execution by "
                "default; rerun only with an explicit debug override for probes "
                "or smoke execution."
            ),
            container_user_space_validated=False,
            native_host_validated=False,
        )

    if status is MatrixCompatibilityStatus.PYTORCH_WHEEL_UNAVAILABLE:
        return MatrixExecutionDecision(
            status=status,
            reason_code=reason_code,
            benchmark_allowed=False,
            probes_allowed=True,
            smoke_allowed=False,
            reason=(
                "Requested dependency stack is unavailable for this Target; this "
                "is a compatibility classification, not a benchmark correctness "
                "failure."
            ),
            container_user_space_validated=False,
            native_host_validated=False,
        )

    if status is MatrixCompatibilityStatus.RUNTIME_UNAVAILABLE:
        return MatrixExecutionDecision(
            status=status,
            reason_code=reason_code,
            benchmark_allowed=False,
            probes_allowed=True,
            smoke_allowed=False,
            reason=(
                "Required ROCm runtime or device access is unavailable; this is a "
                "runtime compatibility classification, not a benchmark correctness "
                "failure."
            ),
            container_user_space_validated=False,
            native_host_validated=False,
        )

    if status is MatrixCompatibilityStatus.NOT_TESTED:
        return MatrixExecutionDecision(
            status=status,
            reason_code=reason_code,
            benchmark_allowed=False,
            probes_allowed=True,
            smoke_allowed=False,
            reason=(
                "Target is not tested and remains non-authoritative; benchmark "
                "eligibility is not implied."
            ),
            container_user_space_validated=False,
            native_host_validated=False,
        )

    return MatrixExecutionDecision(
        status=status,
        reason_code=reason_code,
        benchmark_allowed=True,
        probes_allowed=True,
        smoke_allowed=True,
        reason=entry.reason,
        container_user_space_validated=claims.container_user_space_validated,
        native_host_validated=claims.native_host_validated,
    )

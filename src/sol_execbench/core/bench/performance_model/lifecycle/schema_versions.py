# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Current diagnostic lifecycle and publication artifact schemas."""

from enum import StrEnum


class DiagnosticLifecycleSchema(StrEnum):
    """Canonical diagnostic control-plane and publication identifiers."""

    DIAGNOSTIC_LIFECYCLE_MANIFEST = (
        "sol_execbench.diagnostic_lifecycle_manifest.v1"
    )
    DIAGNOSTIC_LIFECYCLE_STATE = "sol_execbench.diagnostic_lifecycle_state.v1"
    DIAGNOSTIC_PUBLICATION_PROJECTION = (
        "sol_execbench.diagnostic_publication_projection.v2"
    )
    DIAGNOSTIC_PUBLISHED_RELEASE = (
        "sol_execbench.diagnostic_published_release.v3"
    )
    DIAGNOSTIC_RELEASE_PACKAGE = "sol_execbench.diagnostic_release_package.v1"


class DiagnosticLifecycleStateKind(StrEnum):
    """Internal state objects stored by the diagnostic lifecycle engine."""

    ARTIFACT_TREE = "artifact_tree"
    ATTEMPT = "attempt"
    PLAN = "plan"
    RUN = "run"
    STAGE_RECEIPT = "stage_receipt"


class DiagnosticReleasePackageArtifactKind(StrEnum):
    """Artifacts emitted by diagnostic release packaging."""

    ARCHIVE = "archive"
    ATTESTATION = "attestation"


__all__ = [
    "DiagnosticLifecycleSchema",
    "DiagnosticLifecycleStateKind",
    "DiagnosticReleasePackageArtifactKind",
]

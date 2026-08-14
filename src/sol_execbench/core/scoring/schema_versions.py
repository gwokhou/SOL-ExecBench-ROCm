# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Current official scoring and release artifact schemas."""

from enum import StrEnum


class ReleaseArtifactSchema(StrEnum):
    """Canonical score verification and release identifiers."""

    OFFICIAL_SCORE_AVAILABILITY = "sol_execbench.official_score_availability.v3"
    RELEASE_BUNDLE = "sol_execbench.release_bundle.v2"
    RELEASE_COMPONENT = "sol_execbench.release_component.v1"
    RELEASE_PACKAGE = "sol_execbench.release_package.v2"


class ReleasePackageArtifactKind(StrEnum):
    """Artifacts emitted by official score release packaging."""

    ARCHIVE = "archive"
    ATTESTATION = "attestation"


class ReleaseComponentKind(StrEnum):
    """Content-addressed components consumed by a release bundle."""

    ENVIRONMENT = "environment"
    EXECUTION_PLAN = "execution_plan"
    RUN_STATEMENT = "run_statement"
    SOLAR_INDEX = "solar_index"


__all__ = [
    "ReleaseArtifactSchema",
    "ReleaseComponentKind",
    "ReleasePackageArtifactKind",
]

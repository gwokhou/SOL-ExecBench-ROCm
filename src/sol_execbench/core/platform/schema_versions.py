# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Current platform, environment, and toolchain artifact schemas."""

from enum import StrEnum


class PlatformArtifactSchema(StrEnum):
    """Canonical platform capability and environment identifiers."""

    ARCH_CAPABILITY_BUDGET = "sol_execbench.arch_capability_budget.v1"
    PLATFORM_PREFLIGHT = "sol_execbench.platform_preflight.v1"
    ENVIRONMENT_EVIDENCE = "sol_execbench.environment_evidence.v1"
    RDNA4_VALIDATION = "sol_execbench.rdna4_validation.v2"
    RDNA4_VALIDATION_RECEIPT = "sol_execbench.rdna4_validation_receipt.v1"
    ROCM_COMPATIBILITY_MATRIX = "sol_execbench.rocm_compatibility_matrix.v1"
    ROCM_DOCKER_TARGETS = "sol_execbench.rocm_docker_targets.v1"
    STATIC_TARGET_DESCRIPTOR = "sol_execbench.static_target_descriptor.v1"
    TOOLCHAIN_ROUTING = "sol_execbench.toolchain_routing.v1"


class PlatformPreflightArtifactKind(StrEnum):
    """Shell-facing platform preflight result variants."""

    DEPENDENCY = "dependency"
    DOCKER = "docker"


class EnvironmentEvidenceArtifactKind(StrEnum):
    """Standalone and aggregate environment evidence variants."""

    DIAGNOSTICS = "diagnostics"
    SNAPSHOT = "snapshot"


class RDNA4ValidationArtifactKind(StrEnum):
    """Artifacts emitted by the release hardware-validation gate."""

    RECEIPT = "validation_receipt"


__all__ = [
    "EnvironmentEvidenceArtifactKind",
    "PlatformArtifactSchema",
    "PlatformPreflightArtifactKind",
    "RDNA4ValidationArtifactKind",
]

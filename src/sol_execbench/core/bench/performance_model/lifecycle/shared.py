# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Leaf types shared by lifecycle manifests and stage receipts."""

from __future__ import annotations

from pydantic import Field, field_validator

from sol_execbench.core.bench.performance_model.lifecycle.enums import (
    DiagnosticLifecycleStage,
)
from sol_execbench.core.data.base_model import FrozenArtifactModel
from sol_execbench.core.integrity import (
    SHA256Digest,
    validate_relative_artifact_path,
)


class DiagnosticLifecycleArtifact(FrozenArtifactModel):
    """One regular file in a lifecycle object's exact inventory.

    ``relative_path`` is empty for a root manifest and otherwise a validated
    relative artifact path.
    """

    relative_path: str = ""
    sha256: SHA256Digest
    size_bytes: int = Field(ge=0)

    @field_validator("relative_path")
    @classmethod
    def _validate_relative_path(cls, value: str) -> str:
        if value == "":
            return value
        return validate_relative_artifact_path(value, "lifecycle path")


class DiagnosticLifecycleParent(FrozenArtifactModel):
    """One immutable parent object cited by a lifecycle manifest."""

    stage: DiagnosticLifecycleStage
    stage_id: str = Field(min_length=1)
    sha256: SHA256Digest


class GpuLifecycleIdentity(FrozenArtifactModel):
    """GPU and software identity bound when a lifecycle stage used hardware."""

    gpu_architecture: str = Field(min_length=1)
    gpu_id: str | None = None
    gpu_bdf: str | None = None
    rocm_version: str | None = None
    compiler_version: str | None = None
    clock_mode: str | None = None
    power_profile: str | None = None


class SoftwareLifecycleIdentity(FrozenArtifactModel):
    """Producer software identity for a lifecycle stage."""

    sol_version: str = Field(min_length=1)
    python_version: str | None = None


__all__ = [
    "DiagnosticLifecycleArtifact",
    "DiagnosticLifecycleParent",
    "GpuLifecycleIdentity",
    "SoftwareLifecycleIdentity",
]

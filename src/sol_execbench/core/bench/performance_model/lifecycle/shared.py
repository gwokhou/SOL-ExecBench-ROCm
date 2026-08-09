# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Leaf types shared by lifecycle manifests and stage receipts."""

from __future__ import annotations

from pydantic import Field, field_validator, model_validator

from sol_execbench.core.bench.performance_model.lifecycle.enums import (
    DiagnosticEvidencePurpose,
    DiagnosticLifecycleStage,
)
from sol_execbench.core.data.base_model import FrozenArtifactModel
from sol_execbench.core.integrity import (
    SHA256Digest,
    validate_relative_artifact_path,
)
from sol_execbench.core.platform.runtime import PCIeTopologyIdentity


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
    purpose: DiagnosticEvidencePurpose = DiagnosticEvidencePurpose.PRODUCTION
    stage_id: str = Field(min_length=1)
    sha256: SHA256Digest


class GpuLifecycleIdentity(FrozenArtifactModel):
    """GPU and software identity bound when a lifecycle stage used hardware."""

    gpu_architecture: str = Field(min_length=1)
    gpu_id: str | None = None
    gpu_bdf: str | None = None
    pcie_topology: PCIeTopologyIdentity | None = None
    rocm_version: str | None = None
    compiler_version: str | None = None
    clock_mode: str | None = None
    power_profile: str | None = None

    @model_validator(mode="after")
    def _topology_terminates_at_gpu(self) -> GpuLifecycleIdentity:
        if self.pcie_topology is None:
            return self
        if self.gpu_bdf != self.pcie_topology.links[-1].bdf:
            raise ValueError("PCIe topology does not terminate at gpu_bdf")
        return self


class SoftwareLifecycleIdentity(FrozenArtifactModel):
    """Producer software identity for a lifecycle stage."""

    sol_version: str = Field(min_length=1)
    python_version: str | None = None


def require_complete_gpu_identity(
    gpu: GpuLifecycleIdentity | None,
    *,
    stage: DiagnosticLifecycleStage,
    require_pcie_topology: bool = False,
) -> None:
    """When a stage binds hardware, every fingerprint field must be set.

    ``None`` is the only acceptable value for stages that never touch a GPU.
    A partial fingerprint — any of ``gpu_id``, ``gpu_bdf``, ``rocm_version``,
    ``compiler_version``, ``clock_mode``, ``power_profile`` missing — is not
    an authoritative production identity. Calibration, collection, and model
    build stages that use hardware must therefore carry a complete, immutable
    GPU fingerprint that participates in the stage identity.
    """
    if gpu is None:
        if require_pcie_topology:
            raise ValueError(
                f"{stage.value} requires a complete gpu_identity",
            )
        return
    required = (
        "gpu_id",
        "gpu_bdf",
        "rocm_version",
        "compiler_version",
        "clock_mode",
        "power_profile",
    )
    missing = [name for name in required if getattr(gpu, name) is None]
    if require_pcie_topology and gpu.pcie_topology is None:
        missing.append("pcie_topology")
    if missing:
        raise ValueError(
            f"{stage.value} bound hardware but gpu_identity is missing "
            f"required fields: {', '.join(missing)}",
        )


__all__ = [
    "DiagnosticLifecycleArtifact",
    "DiagnosticLifecycleParent",
    "GpuLifecycleIdentity",
    "SoftwareLifecycleIdentity",
    "require_complete_gpu_identity",
]

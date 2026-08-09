# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Cross-artifact hardware identity for lifecycle calibration admission."""

from __future__ import annotations

from pathlib import Path

from sol_execbench.core.bench.performance_model.calibration_audit import (
    DiagnosticCalibrationAudit,
)
from sol_execbench.core.bench.performance_model.lifecycle.enums import (
    DiagnosticEvidencePurpose,
    DiagnosticLifecycleStage,
)
from sol_execbench.core.bench.performance_model.lifecycle.shared import (
    GpuLifecycleIdentity,
    require_complete_gpu_identity,
)
from sol_execbench.core.bench.performance_model.models import (
    DiagnosticCalibrationProfile,
)
from sol_execbench.core.data.json_utils import load_json_file


def load_calibration_gpu_identity(
    profile_path: Path,
    audit_path: Path,
    *,
    expected_purpose: DiagnosticEvidencePurpose,
    require_pcie_topology: bool,
) -> GpuLifecycleIdentity:
    """Load matching calibration artifacts and return their GPU identity."""
    profile = load_json_file(DiagnosticCalibrationProfile, profile_path)
    audit = load_json_file(DiagnosticCalibrationAudit, audit_path)
    if profile.purpose is not expected_purpose or audit.purpose is not (
        expected_purpose
    ):
        raise ValueError("calibration evidence purpose mismatch")
    identity = profile.identity
    probe = audit.probe_identity
    observed = (
        probe.architecture,
        probe.gpu_id,
        probe.gpu_bdf,
        probe.pcie_topology,
        probe.rocm_version,
        probe.compiler_version,
    )
    expected = (
        identity.gpu_architecture,
        identity.gpu_id,
        identity.gpu_bdf,
        identity.pcie_topology,
        identity.rocm_version,
        identity.compiler_version,
    )
    if observed != expected:
        raise ValueError("calibration profile/audit GPU identity mismatch")
    gpu = GpuLifecycleIdentity(**identity.model_dump(mode="python"))
    require_complete_gpu_identity(
        gpu,
        stage=DiagnosticLifecycleStage.CALIBRATION,
        require_pcie_topology=require_pcie_topology,
    )
    return gpu


__all__ = ["load_calibration_gpu_identity"]

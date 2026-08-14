# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Canonical loading and audit verification for calibration profiles."""

from pathlib import Path

from sol_execbench.core.bench.performance_model.calibration_audit import (
    DiagnosticCalibrationAudit,
    calibration_probe_identity_payload,
)
from sol_execbench.core.bench.performance_model.models import (
    DiagnosticCalibrationProfile,
)
from sol_execbench.core.data.json_utils import load_json_file
from sol_execbench.core.integrity import sha256_file, stable_json_checksum


def load_calibration_profile(
    path: Path | None,
) -> DiagnosticCalibrationProfile | None:
    """Load a profile and require its adjacent audit to verify exactly."""
    if path is None:
        return None
    profile = load_json_file(DiagnosticCalibrationProfile, path)
    audit_path = path.with_name(f"{path.stem}.audit.json")
    if not audit_path.is_file():
        raise ValueError("calibration_audit_missing")
    audit = load_json_file(DiagnosticCalibrationAudit, audit_path)
    _verify_calibration_audit(profile, audit, audit_path)
    return profile


def _verify_calibration_audit(
    profile: DiagnosticCalibrationProfile,
    audit: DiagnosticCalibrationAudit,
    audit_path: Path,
) -> None:
    probe = audit.probe_identity
    protocol = audit.protocol
    tuning = [item.model_dump(mode="json") for item in audit.tuning_evidence]
    estimation = [
        item.model_dump(mode="json")
        for item in audit.parameter_estimation_evidence
    ]
    estimation_hashes = {
        stable_json_checksum(estimation),
        sha256_file(audit_path),
    }
    if not estimation_hashes <= set(
        profile.parameter_estimation_evidence_sha256
    ):
        raise ValueError("calibration_parameter_estimation_sha256_mismatch")
    if stable_json_checksum(tuning) not in profile.tuning_evidence_sha256:
        raise ValueError("calibration_tuning_evidence_sha256_mismatch")
    if (
        stable_json_checksum(calibration_probe_identity_payload(probe))
        not in profile.probe_evidence_sha256
    ):
        raise ValueError("calibration_probe_evidence_sha256_mismatch")
    if (
        probe.architecture != profile.identity.gpu_architecture
        or probe.rocm_version != profile.identity.rocm_version
        or probe.gpu_id != profile.identity.gpu_id
        or probe.gpu_bdf != profile.identity.gpu_bdf
        or probe.compiler_version != profile.identity.compiler_version
    ):
        raise ValueError("calibration_audit_identity_mismatch")
    if not protocol.configuration_frozen_before_parameter_estimation or any(
        batch.get("phase") != "parameter_estimation_after_configuration_freeze"
        or batch.get("clocks_locked") is not True
        for batch in estimation
    ):
        raise ValueError("calibration_parameter_estimation_protocol_invalid")
    if (
        protocol.bootstrap_seed != profile.bootstrap_seed
        or protocol.bootstrap_replicates != profile.bootstrap_replicates
    ):
        raise ValueError("calibration_bootstrap_protocol_mismatch")
    if protocol.parameter_estimation_process_batches < 5:
        raise ValueError(
            "calibration_parameter_estimation_processes_insufficient"
        )


__all__ = ["load_calibration_profile"]

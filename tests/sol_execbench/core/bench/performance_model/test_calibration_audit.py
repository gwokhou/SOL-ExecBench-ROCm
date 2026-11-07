from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest
from pydantic import ValidationError

from sol_execbench.core.bench.performance_model.calibration_audit import (
    CalibrationMetric,
    CalibrationProbeBatch,
    CalibrationProbeIdentity,
    CalibrationProtocol,
    DiagnosticCalibrationAudit,
    calibration_probe_identity_payload,
)
from sol_execbench.core.bench.performance_model.lifecycle.calibration_identity import (
    load_calibration_gpu_identity,
)
from sol_execbench.core.bench.performance_model.lifecycle.enums import (
    DiagnosticEvidencePurpose,
)
from sol_execbench.core.bench.performance_model.models import (
    CalibrationIdentity,
    CalibrationParameter,
    CalibrationParameterName,
    CalibrationUnit,
    DiagnosticCalibrationProfile,
)
from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.evidence.runtime_evidence.models import (
    RuntimeGPUTelemetry,
)
from sol_execbench.core.platform.hardware import (
    PCIeLinkIdentity,
    PCIeTopologyIdentity,
)


def _topology(width: int = 8) -> PCIeTopologyIdentity:
    link = PCIeLinkIdentity(
        bdf="0000:03:00.0",
        current_speed_gtps=32.0,
        max_speed_gtps=32.0,
        current_width=width,
        max_width=16,
    )
    return PCIeTopologyIdentity(
        links=(link,),
        bottleneck_bdf=link.bdf,
        effective_speed_gtps=link.current_speed_gtps,
        effective_width=link.current_width,
    )


def _telemetry(
    phase: Literal["pre", "post"],
    topology: PCIeTopologyIdentity | None,
) -> RuntimeGPUTelemetry:
    return RuntimeGPUTelemetry.model_validate(
        {
            "phase": phase,
            "gpu_id": "gpu-0",
            "gpu_bdf": "0000:03:00.0",
            "pcie_topology": topology,
            "performance_level": "AMDSMI_DEV_PERF_LEVEL_STABLE_PEAK",
            "temperature_c": 50.0,
            "foreign_process_count": 0,
        }
    )


def _audit(
    pre: PCIeTopologyIdentity | None,
    post: PCIeTopologyIdentity | None,
) -> DiagnosticCalibrationAudit:
    metric = CalibrationMetric(
        name="dispatch_floor",
        variant="default",
        value=1.0,
        unit="ns",
    )
    batch = CalibrationProbeBatch(
        phase="tuning",
        process_batch=0,
        mode="default",
        clocks_locked=True,
        metrics=[metric],
    )
    return DiagnosticCalibrationAudit(
        probe_identity=CalibrationProbeIdentity(
            source_sha256="a" * 64,
            binary_sha256="b" * 64,
            compiler_sha256="c" * 64,
            architecture="gfx1200",
            rocm_version="7.2.0",
            device_name="RX 9060 XT",
            gpu_id="gpu-0",
            gpu_bdf="0000:03:00.0",
            pcie_topology=pre,
            total_memory_bytes=16 * 1024**3,
            compiler_version="HIP 7.2",
            isa={},
        ),
        protocol=CalibrationProtocol(
            design="two_phase_tuning_then_parameter_estimation",
            configuration_frozen_before_parameter_estimation=True,
            tuning_process_batches=1,
            parameter_estimation_process_batches=5,
            bootstrap_replicates=100,
            bootstrap_seed=1,
            clock_mode="STABLE_PEAK",
        ),
        frozen_configuration={"mode": "default"},
        tuning_evidence=[batch],
        parameter_estimation_evidence=[
            batch.model_copy(update={"phase": "parameter_estimation"})
        ],
        environment=[_telemetry("pre", pre), _telemetry("post", post)],
    )


def test_calibration_audit_binds_stable_pcie_topology() -> None:
    topology = _topology()

    assert _audit(topology, topology).probe_identity.pcie_topology == topology


def test_calibration_audit_rejects_pcie_topology_drift() -> None:
    with pytest.raises(ValidationError, match="PCIe topology changed"):
        _audit(_topology(8), _topology(4))


def test_calibration_audit_rejects_incomplete_pcie_topology() -> None:
    with pytest.raises(ValidationError, match="PCIe topology is incomplete"):
        _audit(_topology(), None)


def test_probe_identity_hash_payload_is_additive_for_pcie() -> None:
    legacy = _audit(None, None).probe_identity
    current = _audit(_topology(), _topology()).probe_identity

    assert "pcie_topology" not in calibration_probe_identity_payload(legacy)
    assert calibration_probe_identity_payload(current)["pcie_topology"] == (
        _topology().model_dump(mode="json")
    )


def _profile(
    topology: PCIeTopologyIdentity | None,
) -> DiagnosticCalibrationProfile:
    return DiagnosticCalibrationProfile(
        identity=CalibrationIdentity(
            gpu_architecture="gfx1200",
            gpu_id="gpu-0",
            gpu_bdf="0000:03:00.0",
            pcie_topology=topology,
            rocm_version="7.2.0",
            compiler_version="HIP 7.2",
            clock_mode="locked",
            power_profile="stable_peak",
        ),
        parameters=[
            CalibrationParameter(
                name=CalibrationParameterName.DISPATCH_FLOOR_MS,
                value=0.01,
                unit=CalibrationUnit.MS,
                confidence_interval=(0.009, 0.011),
            )
        ],
        tuning_evidence_sha256=["d" * 64],
        parameter_estimation_evidence_sha256=["e" * 64],
        probe_evidence_sha256=["f" * 64],
        bootstrap_seed=1,
        bootstrap_replicates=100,
    )


def test_production_calibration_identity_requires_pcie_topology(
    tmp_path: Path,
) -> None:
    profile_path = tmp_path / "profile.json"
    audit_path = tmp_path / "audit.json"
    atomic_write_json_value(
        profile_path, _profile(None).model_dump(mode="json")
    )
    atomic_write_json_value(
        audit_path, _audit(None, None).model_dump(mode="json")
    )

    with pytest.raises(ValueError, match="pcie_topology"):
        load_calibration_gpu_identity(
            profile_path,
            audit_path,
            expected_purpose=DiagnosticEvidencePurpose.PRODUCTION,
            require_pcie_topology=True,
        )


def test_production_calibration_identity_matches_profile_and_audit(
    tmp_path: Path,
) -> None:
    topology = _topology()
    profile_path = tmp_path / "profile.json"
    audit_path = tmp_path / "audit.json"
    atomic_write_json_value(
        profile_path,
        _profile(topology).model_dump(mode="json"),
    )
    atomic_write_json_value(
        audit_path,
        _audit(topology, topology).model_dump(mode="json"),
    )

    identity = load_calibration_gpu_identity(
        profile_path,
        audit_path,
        expected_purpose=DiagnosticEvidencePurpose.PRODUCTION,
        require_pcie_topology=True,
    )

    assert identity.pcie_topology == topology

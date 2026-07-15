from __future__ import annotations

import json
import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from sol_execbench.cli.commands.hardware_model import _validate_shape_aware_evidence
from sol_execbench.core.scoring.hardware_calibration.shape_aware_roofline import (
    ShapeAwareRooflineArtifact,
    ShapeAwareRooflineCase,
    shape_aware_roofline_from_dict,
    validate_shape_aware_raw_evidence,
)
from sol_execbench.core.scoring.hardware_profile_requirements import (
    HardwareProfileRequirements,
)
from sol_execbench.core.integrity.checksums import sha256_file


_CALIBRATION_SHA256S = (
    "1" * 64,
    "2" * 64,
)
_COVERAGE_SHA256 = "4" * 64
_PLAN_SHA256 = "6" * 64
_COLLECTION_REPORT_SHA256S = ("7" * 64, "8" * 64)
_PROFILE = "compute.vector.fp32.fp32.gfx12"


def _artifact(
    requirements_sha256: str | None = "3" * 64,
    coverage_sha256: str = _COVERAGE_SHA256,
) -> ShapeAwareRooflineArtifact:
    assert requirements_sha256 is not None
    return ShapeAwareRooflineArtifact(
        generated_at="2026-07-13T00:00:00Z",
        architecture="gfx1200",
        calibration_sha256s=_CALIBRATION_SHA256S,
        requirements_sha256=requirements_sha256,
        authority_coverage_sha256=coverage_sha256,
        plan_payload_sha256=_PLAN_SHA256,
        collection_report_sha256s=_COLLECTION_REPORT_SHA256S,
        bucketing_dimensions=("shape", "layout", "launch", "occupancy"),
        cases=(
            ShapeAwareRooflineCase(
                case_id="vector-contiguous-large",
                profile_key=_PROFILE,
                shape=(1 << 20,),
                layout="contiguous",
                launch={"grid_x": 4096, "block_x": 256},
                occupancy={"waves_per_workgroup": 4, "workgroups_per_cu": 8},
                warmup_iterations=3,
                samples_ms=(1.0,) * 7,
                covered_workloads=(("first", "w1", "L1/first"),),
                raw_evidence_ref="raw/vector-contiguous-large.json",
                raw_evidence_sha256="5" * 64,
            ),
        ),
        collection_status="collected",
        validation_status="validated",
    )


def test_shape_aware_artifact_round_trip_and_model_build_binding(tmp_path) -> None:
    requirements = HardwareProfileRequirements(
        architecture="gfx1200",
        required_profile_keys=(_PROFILE,),
        scope="fixture",
    )
    raw_path = tmp_path / "raw" / "vector-contiguous-large.json"
    raw_path.parent.mkdir()
    provider_path = tmp_path / "provider.json"
    trace_path = tmp_path / "trace.csv"
    provider_path.write_text("provider\n", encoding="utf-8")
    trace_path.write_text("trace\n", encoding="utf-8")
    raw_payload = {
        "schema_version": "sol_execbench.shape_aware_roofline_raw.v1",
        "created_at": "2026-07-13T00:00:00Z",
        "architecture": "gfx1200",
        "definition": "first",
        "workload_uuid": "w1",
        "problem_id": "L1/first",
        "profile_keys": [_PROFILE],
        "shape": [1 << 20],
        "layout": "contiguous",
        "tensor_layouts": {"inputs": [], "outputs": []},
        "samples_ms": [1.0] * 7,
        "provider_evidence_ref": str(provider_path),
        "provider_evidence_sha256": sha256_file(provider_path),
        "trace_files": [{"path": str(trace_path), "sha256": sha256_file(trace_path)}],
        "launch": {"grid_x": 4096, "block_x": 256},
        "kernel_resources": {},
        "occupancy_counters": {"OccupancyPercent": 50.0},
        "representative_dispatch_duration_ns": 1,
        "occupancy_status": "measured",
    }
    raw_payload["payload_sha256"] = hashlib.sha256(
        json.dumps(
            raw_payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    raw_path.write_text(json.dumps(raw_payload), encoding="utf-8")
    case = replace(
        _artifact().cases[0],
        raw_evidence_ref=str(raw_path),
        raw_evidence_sha256=sha256_file(raw_path),
    )
    artifact = replace(
        _artifact(requirements.payload_sha256), cases=(case,), payload_sha256=None
    )
    parsed = shape_aware_roofline_from_dict(artifact.to_dict())

    assert parsed.to_dict() == artifact.to_dict()
    _validate_shape_aware_evidence(
        parsed,
        evidence_path=tmp_path / "shape-aware.json",
        architecture="gfx1200",
        primary_sha256=_CALIBRATION_SHA256S[0],
        verification_sha256=_CALIBRATION_SHA256S[1],
        requirements=requirements,
        authority_coverage_sha256=_COVERAGE_SHA256,
    )


def test_shape_aware_artifact_rejects_unstable_or_unbound_evidence() -> None:
    with pytest.raises(ValueError, match="spread"):
        ShapeAwareRooflineCase(
            case_id="unstable",
            profile_key=_PROFILE,
            shape=(1024,),
            layout="contiguous",
            launch={"grid_x": 4, "block_x": 256},
            occupancy={"waves_per_workgroup": 4},
            warmup_iterations=3,
            samples_ms=(1.0, 1.0, 1.1, 2.0, 2.0, 2.0, 2.0),
            covered_workloads=(("first", "w1", "L1/first"),),
            raw_evidence_ref="raw.json",
            raw_evidence_sha256="5" * 64,
        )
    requirements = HardwareProfileRequirements(
        architecture="gfx1200",
        required_profile_keys=("memory.stream_copy.fp32.fp32.gfx12",),
        scope="fixture",
    )
    evidence = _artifact(requirements.payload_sha256)
    with pytest.raises(ValueError, match="lacks required profile"):
        _validate_shape_aware_evidence(
            evidence,
            evidence_path=Path("unused.json"),
            architecture="gfx1200",
            primary_sha256=_CALIBRATION_SHA256S[0],
            verification_sha256=_CALIBRATION_SHA256S[1],
            requirements=requirements,
            authority_coverage_sha256=_COVERAGE_SHA256,
        )


def test_shape_aware_raw_evidence_rejects_dispatch_count_as_occupancy(tmp_path) -> None:
    requirements = HardwareProfileRequirements(
        architecture="gfx1200",
        required_profile_keys=(_PROFILE,),
        scope="fixture",
    )
    case = _artifact(requirements.payload_sha256).cases[0]
    provider_path = tmp_path / "provider.json"
    trace_path = tmp_path / "trace.csv"
    provider_path.write_text("provider\n", encoding="utf-8")
    trace_path.write_text("trace\n", encoding="utf-8")
    raw_path = tmp_path / "raw.json"
    raw_payload = {
        "schema_version": "sol_execbench.shape_aware_roofline_raw.v1",
        "created_at": "2026-07-13T00:00:00Z",
        "architecture": "gfx1200",
        "definition": "first",
        "workload_uuid": "w1",
        "problem_id": "L1/first",
        "profile_keys": [_PROFILE],
        "shape": [1 << 20],
        "layout": "contiguous",
        "tensor_layouts": [],
        "samples_ms": [1.0] * 7,
        "provider_evidence_ref": str(provider_path),
        "provider_evidence_sha256": sha256_file(provider_path),
        "trace_files": [{"path": str(trace_path), "sha256": sha256_file(trace_path)}],
        "launch": {"grid_x": 4096, "block_x": 256},
        "kernel_resources": {},
        "occupancy_counters": {"SQ_WAVES": 896.0, "SQ_BUSY_CYCLES": 100.0},
        "representative_dispatch_duration_ns": 1,
        "occupancy_status": "measured",
    }
    raw_payload["payload_sha256"] = hashlib.sha256(
        json.dumps(raw_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    raw_path.write_text(json.dumps(raw_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="occupancy counters are invalid"):
        validate_shape_aware_raw_evidence(case, raw_path, architecture="gfx1200")


def test_model_build_rejects_missing_raw_shape_evidence(tmp_path) -> None:
    requirements = HardwareProfileRequirements(
        architecture="gfx1200", required_profile_keys=(_PROFILE,), scope="fixture"
    )
    evidence = _artifact(requirements.payload_sha256)
    with pytest.raises(ValueError, match="raw evidence is missing"):
        _validate_shape_aware_evidence(
            evidence,
            evidence_path=tmp_path / "shape-aware.json",
            architecture="gfx1200",
            primary_sha256=_CALIBRATION_SHA256S[0],
            verification_sha256=_CALIBRATION_SHA256S[1],
            requirements=requirements,
            authority_coverage_sha256=_COVERAGE_SHA256,
        )


def test_model_build_writes_only_checksum_bound_shape_evidence(
    tmp_path, monkeypatch
) -> None:
    from sol_execbench.cli.commands import hardware_model
    from sol_execbench.core.scoring.hardware_calibration.environment import (
        GpuEnvironment,
    )
    from sol_execbench.core.scoring.hardware_calibration.models import (
        CalibrationCandidate,
        HardwareCalibrationArtifact,
    )

    requirements = HardwareProfileRequirements(
        architecture="gfx1200", required_profile_keys=(_PROFILE,), scope="fixture"
    )
    coverage_path = tmp_path / "authority-coverage.json"
    coverage_path.write_text('{"schema_version":"fixture"}\n', encoding="utf-8")
    evidence = _artifact(requirements.payload_sha256, sha256_file(coverage_path))
    evidence_path = tmp_path / "shape-aware.json"
    evidence_path.write_text(json.dumps(evidence.to_dict()), encoding="utf-8")
    requirements_path = tmp_path / "requirements.json"
    requirements_path.write_text(json.dumps(requirements.to_dict()), encoding="utf-8")

    def calibration(generated_at: str) -> HardwareCalibrationArtifact:
        return HardwareCalibrationArtifact(
            generated_at=generated_at,
            candidates=(
                CalibrationCandidate(_PROFILE, "measured", 5.0, "TFLOP/s", (5.0,) * 7),
            ),
            collection_status="collected",
            validation_status="validated",
            metadata={
                "device": 0,
                "gpu_uuid": "GPU-a",
                "architecture": "gfx1200",
                "rocm_version": "7.1.1",
                "adapter_policy": {"requires_clock_lock": True},
                "clock_observations": {"pre": False, "during": True, "post": False},
                "profile_requirements": {
                    "required_profile_keys": [_PROFILE],
                    "requirements_sha256": requirements.payload_sha256,
                },
                "probe_protocol": {"warmup_iterations": 3, "timed_samples": 7},
                "probe_evidence": {
                    _PROFILE: {
                        "compiler_command": ["hipcc"],
                        "source_sha256": "6" * 64,
                        "binary_sha256": "7" * 64,
                        "warmup_iterations": 3,
                        "timed_samples": 7,
                    }
                },
            },
        )

    primary, verification = (
        calibration("2026-07-13T00:00:00Z"),
        calibration("2026-07-13T00:01:00Z"),
    )
    raw_path = tmp_path / "raw" / "vector-contiguous-large.json"
    raw_path.parent.mkdir()
    provider_path = tmp_path / "integration-provider.json"
    trace_path = tmp_path / "integration-trace.csv"
    provider_path.write_text("provider\n", encoding="utf-8")
    trace_path.write_text("trace\n", encoding="utf-8")
    raw_payload = {
        "schema_version": "sol_execbench.shape_aware_roofline_raw.v1",
        "created_at": "2026-07-13T00:00:00Z",
        "architecture": "gfx1200",
        "definition": "first",
        "workload_uuid": "w1",
        "problem_id": "L1/first",
        "profile_keys": [_PROFILE],
        "shape": [1 << 20],
        "layout": "contiguous",
        "tensor_layouts": {"inputs": [], "outputs": []},
        "samples_ms": [1.0] * 7,
        "provider_evidence_ref": str(provider_path),
        "provider_evidence_sha256": sha256_file(provider_path),
        "trace_files": [{"path": str(trace_path), "sha256": sha256_file(trace_path)}],
        "launch": {"grid_x": 4096, "block_x": 256},
        "kernel_resources": {},
        "occupancy_counters": {"OccupancyPercent": 50.0},
        "representative_dispatch_duration_ns": 1,
        "occupancy_status": "measured",
    }
    raw_payload["payload_sha256"] = hashlib.sha256(
        json.dumps(
            raw_payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    raw_path.write_text(json.dumps(raw_payload), encoding="utf-8")
    plan = {
        "schema_version": "sol_execbench.shape_aware_roofline_plan.v1",
        "architecture": "gfx1200",
        "authority_coverage_sha256": sha256_file(coverage_path),
        "requirements_sha256": requirements.payload_sha256,
        "required_dimensions": ["shape", "layout", "launch", "occupancy"],
        "authority_workload_count": 1,
        "profile_shards": [
            {
                "profile_key": _PROFILE,
                "workloads": [
                    {
                        "definition": "first",
                        "workload_uuid": "w1",
                        "problem_id": "L1/first",
                    }
                ],
            }
        ],
    }
    plan["payload_sha256"] = hashlib.sha256(
        json.dumps(
            plan, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    plan_path = tmp_path / "shape-aware-plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    # Recreate envelope after scalar checksum values are available.
    evidence = replace(
        _artifact(requirements.payload_sha256, sha256_file(coverage_path)),
        cases=(
            replace(
                _artifact().cases[0],
                raw_evidence_sha256=sha256_file(raw_path),
            ),
        ),
        calibration_sha256s=(primary.payload_sha256, verification.payload_sha256),
        plan_payload_sha256=plan["payload_sha256"],
        payload_sha256=None,
    )
    evidence_path.write_text(json.dumps(evidence.to_dict()), encoding="utf-8")
    primary_path, verification_path = (
        tmp_path / "primary.json",
        tmp_path / "verify.json",
    )
    primary_path.write_text(json.dumps(primary.to_dict()), encoding="utf-8")
    verification_path.write_text(json.dumps(verification.to_dict()), encoding="utf-8")
    monkeypatch.setattr(
        hardware_model,
        "discover_gpu",
        lambda _: GpuEnvironment(0, "gfx1200", uuid="GPU-a", rocm_version="7.1.1"),
    )

    output = tmp_path / "model.json"
    callback = hardware_model._build.callback
    assert callback is not None
    callback(
        primary_path,
        output,
        None,
        verification_path,
        requirements_path,
        evidence_path,
        coverage_path,
        plan_path,
    )
    model = json.loads(output.read_text(encoding="utf-8"))
    assert model["shape_aware_roofline"]["evidence_refs"] == [
        f"{evidence_path}#sha256:{evidence.payload_sha256}"
    ]

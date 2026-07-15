"""CLI tests for calibrated AMD hardware-model evidence."""

from __future__ import annotations

import json

import pytest

from click.testing import CliRunner

from sol_execbench.cli.main import cli
from sol_execbench.core.scoring.hardware_calibration.models import (
    CalibrationCandidate,
    HardwareCalibrationArtifact,
)


def test_build_authority_rejects_mismatched_live_environment() -> None:
    from sol_execbench.cli.commands.hardware_model import (
        _validate_calibration_authority,
    )
    from sol_execbench.core.scoring.hardware_calibration.environment import (
        GpuEnvironment,
    )

    artifact = HardwareCalibrationArtifact(
        generated_at="2026-07-10T00:00:00Z",
        candidates=(
            CalibrationCandidate(
                "compute.vector.fp32.fp32.portable",
                "measured",
                5.0,
                "TFLOP/s",
                (5.0,) * 7,
            ),
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
                "required_profile_keys": ["compute.vector.fp32.fp32.portable"]
            },
            "probe_protocol": {"warmup_iterations": 3, "timed_samples": 7},
        },
    )

    with pytest.raises(ValueError, match="GPU UUID"):
        _validate_calibration_authority(
            artifact,
            GpuEnvironment(0, "gfx1200", uuid="GPU-b", rocm_version="7.1.1"),
            None,
        )
    with pytest.raises(ValueError, match="architecture"):
        _validate_calibration_authority(
            artifact,
            GpuEnvironment(0, "gfx942", uuid="GPU-a", rocm_version="7.1.1"),
            None,
        )
    with pytest.raises(ValueError, match="ROCm"):
        _validate_calibration_authority(
            artifact,
            GpuEnvironment(0, "gfx1200", uuid="GPU-a", rocm_version="7.2.0"),
            None,
        )


def test_build_profile_evidence_ref_binds_calibration_checksum(
    tmp_path, monkeypatch
) -> None:
    from sol_execbench.cli.commands import hardware_model
    from sol_execbench.core.scoring.hardware_calibration.environment import (
        GpuEnvironment,
    )

    artifact = HardwareCalibrationArtifact(
        generated_at="2026-07-10T00:00:00Z",
        candidates=(
            CalibrationCandidate(
                "compute.vector.fp32.fp32.portable",
                "measured",
                5.0,
                "TFLOP/s",
                (5.0,) * 7,
            ),
            CalibrationCandidate(
                "compute.matrix.bf16.bf16.wmma",
                "measured",
                10.0,
                "TFLOP/s",
                (10.0,) * 7,
            ),
            CalibrationCandidate(
                "compute.matrix.fp16.fp16.wmma",
                "measured",
                10.0,
                "TFLOP/s",
                (10.0,) * 7,
            ),
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
                "required_profile_keys": ["compute.vector.fp32.fp32.portable"]
            },
            "probe_protocol": {"warmup_iterations": 3, "timed_samples": 7},
        },
    )
    calibration_path = tmp_path / "calibration.json"
    output = tmp_path / "model.json"
    calibration_path.write_text(json.dumps(artifact.to_dict()))
    monkeypatch.setattr(
        hardware_model,
        "discover_gpu",
        lambda _: GpuEnvironment(0, "gfx1200", uuid="GPU-a", rocm_version="7.1.1"),
    )

    callback = hardware_model._build.callback
    assert callback is not None
    callback(calibration_path, output, None)

    model = json.loads(output.read_text())
    assert (
        f"sha256:{artifact.payload_sha256}"
        in model["compute_profiles"][0]["evidence_ref"]
    )


def test_build_rejects_measured_mismatched_matrix_evidence(
    tmp_path, monkeypatch
) -> None:
    from sol_execbench.cli.commands import hardware_model
    from sol_execbench.core.scoring.hardware_calibration.environment import (
        GpuEnvironment,
    )

    artifact = HardwareCalibrationArtifact(
        generated_at="2026-07-10T00:00:00Z",
        candidates=(
            CalibrationCandidate(
                "compute.vector.fp32.fp32.portable",
                "measured",
                5.0,
                "TFLOP/s",
                (5.0,) * 7,
            ),
            CalibrationCandidate(
                "compute.matrix.bf16.bf16.mfma",
                "measured",
                10.0,
                "TFLOP/s",
                (10.0,) * 7,
            ),
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
                "required_profile_keys": ["compute.vector.fp32.fp32.portable"]
            },
            "probe_protocol": {"warmup_iterations": 3, "timed_samples": 7},
        },
    )
    calibration_path = tmp_path / "calibration.json"
    output = tmp_path / "model.json"
    calibration_path.write_text(json.dumps(artifact.to_dict()))
    monkeypatch.setattr(
        hardware_model,
        "discover_gpu",
        lambda _: GpuEnvironment(0, "gfx1200", uuid="GPU-a", rocm_version="7.1.1"),
    )

    result = CliRunner().invoke(
        cli,
        [
            "hardware",
            "model",
            "build",
            "--calibration",
            str(calibration_path),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code != 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "rejected"
    assert "compute.matrix.bf16.bf16.wmma" in payload["reason"]


def test_calibrate_writes_rejected_artifact_before_nonzero_exit(
    tmp_path, monkeypatch
) -> None:
    from sol_execbench.cli.commands import hardware_model
    from sol_execbench.core.scoring.hardware_calibration.environment import (
        GpuEnvironment,
    )

    monkeypatch.setattr(
        hardware_model,
        "discover_gpu",
        lambda _: GpuEnvironment(0, "gfx1200"),
    )
    output = tmp_path / "calibration.json"

    result = CliRunner().invoke(
        cli,
        [
            "hardware",
            "model",
            "calibrate",
            "--device",
            "0",
            "--architecture",
            "unsupported-gfx",
            "--output",
            str(output),
            "--offline",
            "--no-auto-install",
        ],
    )

    assert result.exit_code != 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "rejected"
    assert payload["diagnostic"] is True
    assert (
        "architecture assertion does not match runtime discovery" in payload["reason"]
    )


def test_calibrate_rejects_noncanonical_source_revision(tmp_path) -> None:
    output = tmp_path / "calibration.json"

    result = CliRunner().invoke(
        cli,
        [
            "hardware",
            "model",
            "calibrate",
            "--output",
            str(output),
            "--source-revision",
            "not-a-full-git-revision",
            "--offline",
            "--no-auto-install",
        ],
    )

    assert result.exit_code != 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "rejected"
    assert "source revision" in payload["reason"]


def test_build_rejects_noncanonical_source_revision(tmp_path) -> None:
    calibration = tmp_path / "calibration.json"
    calibration.write_text("{}", encoding="utf-8")
    output = tmp_path / "model.json"

    result = CliRunner().invoke(
        cli,
        [
            "hardware",
            "model",
            "build",
            "--calibration",
            str(calibration),
            "--output",
            str(output),
            "--source-revision",
            "not-a-full-git-revision",
        ],
    )

    assert result.exit_code != 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "rejected"
    assert "source revision" in payload["reason"]


def test_calibrate_provisional_result_writes_rejected_diagnostic_before_exit(
    tmp_path, monkeypatch
) -> None:
    from sol_execbench.cli.commands import hardware_model
    from sol_execbench.core.scoring.hardware_calibration.environment import (
        GpuEnvironment,
    )

    artifact = HardwareCalibrationArtifact(
        generated_at="2026-07-10T00:00:00Z",
        candidates=(
            CalibrationCandidate(
                "compute.vector.fp32.fp32.portable",
                "unknown",
                None,
                None,
                reason_code="probe_unavailable",
            ),
        ),
        collection_status="collected",
        validation_status="provisional",
        metadata={
            "architecture": "gfx1200",
            "profile_requirements": {
                "required_profile_keys": ["compute.vector.fp32.fp32.portable"]
            },
            "probe_protocol": {"warmup_iterations": 3, "timed_samples": 7},
        },
    )
    monkeypatch.setattr(
        hardware_model, "ensure_profiler_environment", lambda *a, **k: None
    )
    monkeypatch.setattr(
        hardware_model,
        "discover_gpu",
        lambda _: GpuEnvironment(0, "gfx1200"),
    )
    monkeypatch.setattr(hardware_model, "run_calibration", lambda request: artifact)
    output = tmp_path / "calibration.json"

    result = CliRunner().invoke(
        cli,
        [
            "hardware",
            "model",
            "calibrate",
            "--architecture",
            "gfx1200",
            "--output",
            str(output),
            "--offline",
        ],
    )

    assert result.exit_code != 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "rejected"
    assert payload["diagnostic"] is True
    assert "validation provenance" in payload["reason"]

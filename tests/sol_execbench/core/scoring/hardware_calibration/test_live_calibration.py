"""Marker-gated live calibration evidence smoke test."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from sol_execbench.cli.main import cli
from sol_execbench.cli.commands.hardware_model import _validate_calibration_authority
from sol_execbench.core.scoring.hardware_calibration.environment import discover_gpu
from sol_execbench.core.scoring.hardware_calibration.builder import (
    CalibrationRequest,
    run_calibration,
)
from sol_execbench.core.scoring.hardware_calibration.hip_probe import (
    CalibrationProfileKey,
    HipCommandBackend,
    default_hip_probe,
)
from sol_execbench.core.scoring.hardware_calibration.models import (
    hardware_calibration_artifact_from_dict,
)


@pytest.mark.requires_rdna4
def test_live_rdna4_wmma_calibration_is_measured() -> None:
    environment = discover_gpu(0)
    artifact = run_calibration(
        CalibrationRequest(environment=environment, require_clock_lock=True)
    )
    candidate = next(
        candidate
        for candidate in artifact.candidates
        if candidate.key == "compute.matrix.bf16.bf16.wmma"
    )

    assert candidate.state == "measured"
    assert artifact.validation_status == "validated"


@pytest.mark.requires_cdna3
def test_live_cdna3_mfma_calibration_is_measured() -> None:
    environment = discover_gpu(0)
    candidate = default_hip_probe(
        HipCommandBackend(architecture=environment.architecture)
    ).measure(CalibrationProfileKey("compute", "matrix", "bf16", "bf16", "mfma"))

    assert candidate.state == "measured"


@pytest.mark.requires_rocm
def test_live_cdna4_mfma_calibration_is_measured() -> None:
    environment = discover_gpu(0)
    if not environment.architecture.startswith("gfx95"):
        pytest.skip("test requires an AMD CDNA 4 GPU, such as gfx950")
    candidate = default_hip_probe(
        HipCommandBackend(architecture=environment.architecture)
    ).measure(CalibrationProfileKey("compute", "matrix", "bf16", "bf16", "mfma"))

    assert candidate.state == "measured"


@pytest.mark.requires_rocm
@pytest.mark.requires_rdna4
def test_live_offline_calibration_writes_rdna4_evidence(tmp_path) -> None:
    """The live command preserves actual measurement states without installs."""
    output = tmp_path / "out" / "calibration.json"

    result = CliRunner().invoke(
        cli,
        [
            "hardware-model",
            "calibrate",
            "--device",
            "0",
            "--architecture",
            "gfx1200",
            "--output",
            str(output),
            "--offline",
        ],
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    if result.exit_code == 0:
        artifact = hardware_calibration_artifact_from_dict(payload)
        assert artifact.validation_status == "validated"
        assert all(candidate.state == "measured" for candidate in artifact.candidates)
        _validate_calibration_authority(artifact, discover_gpu(0), None)
    else:
        # A host without observed STABLE_PEAK evidence must remain diagnostic;
        # the command may not upgrade merely because the lock command was attempted.
        assert payload["status"] == "rejected"
        assert payload["diagnostic"] is True

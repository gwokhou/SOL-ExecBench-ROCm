"""CLI tests for calibrated AMD hardware-model evidence."""

from __future__ import annotations

import json

from click.testing import CliRunner

from sol_execbench.cli.main import cli
from sol_execbench.core.scoring.hardware_calibration.models import (
    CalibrationCandidate,
    HardwareCalibrationArtifact,
)


def test_calibrate_writes_rejected_artifact_before_nonzero_exit(tmp_path) -> None:
    output = tmp_path / "calibration.json"

    result = CliRunner().invoke(
        cli,
        [
            "hardware-model",
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
    assert "unsupported GPU architecture" in payload["reason"]


def test_calibrate_provisional_result_writes_rejected_diagnostic_before_exit(
    tmp_path, monkeypatch
) -> None:
    from sol_execbench.cli.commands import hardware_model

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
        metadata={"architecture": "gfx1200"},
    )
    monkeypatch.setattr(
        hardware_model, "ensure_profiler_environment", lambda *a, **k: None
    )
    monkeypatch.setattr(hardware_model, "run_calibration", lambda request: artifact)
    output = tmp_path / "calibration.json"

    result = CliRunner().invoke(
        cli,
        [
            "hardware-model",
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

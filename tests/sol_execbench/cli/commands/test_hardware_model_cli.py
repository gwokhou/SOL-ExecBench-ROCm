"""CLI tests for calibrated AMD hardware-model evidence."""

from __future__ import annotations

import json

from click.testing import CliRunner

from sol_execbench.cli.main import cli


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

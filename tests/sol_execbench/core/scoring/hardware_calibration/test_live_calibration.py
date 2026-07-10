"""Marker-gated live calibration evidence smoke test."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from sol_execbench.cli.main import cli


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

    # A host without observed STABLE_PEAK evidence must remain diagnostic; the
    # command may not upgrade merely because the lock command was attempted.
    assert result.exit_code != 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "rejected"
    assert payload["diagnostic"] is True

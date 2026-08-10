"""Contracts for capacity-governed gfx1200 diagnostic calibration."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


def test_indexed_read_probe_uses_frozen_capacity_tier(
    load_script: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = load_script(
        "scripts/internal/rdna4/run_rdna4_diagnostic_calibration.py"
    )
    commands: list[list[str]] = []

    def run(command: list[str], *, timeout: float) -> SimpleNamespace:
        del timeout
        commands.append(command)
        return SimpleNamespace(
            returncode=0,
            stdout="METRIC indexed_read_item_per_ms cell 1 item/ms\n",
            stderr="",
        )

    monkeypatch.setattr(script, "run_in_process_group_bounded", run)
    monkeypatch.setattr(
        script, "verify_clock_state_with_warning", lambda **_: True
    )

    batch = script._run_probe_batch(
        Path("/probe"),
        phase="qualification_canary",
        process_batch=0,
        mode="indexed_read",
        vram_policy=SimpleNamespace(probe_working_set_bytes=512 * 2**20),
    )

    assert commands == [["/probe", "indexed_read", "536870912"]]
    assert batch.mode == "indexed_read"
    assert "indexed_read" in script._QUALIFICATION_CANARY_MODES

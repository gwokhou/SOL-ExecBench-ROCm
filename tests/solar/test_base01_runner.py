from __future__ import annotations

import subprocess

import pytest

from scripts.internal.solar import base01_records, run_base01_cpu
from scripts.internal.solar.base01_observations import ReceiptFailure
from scripts.internal.solar.base01_samples import SAMPLES
from solar.ir.contracts import IRPath


@pytest.mark.parametrize(
    "outcome", ["crash", "timeout"], ids=["abnormal-exit", "timed-out"]
)
def test_slot_retains_child_failures_and_the_original_denominator(
    tmp_path, monkeypatch, outcome
):
    def run(command, **kwargs):
        assert kwargs["timeout"] == run_base01_cpu.SLOT_TIMEOUT_SECONDS
        assert kwargs["max_capture_bytes"] == run_base01_cpu.LOG_LIMIT_BYTES
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(
                command, 180, output="started", stderr="unfinished"
            )
        return subprocess.CompletedProcess(command, 7, "started", "crashed")

    monkeypatch.setattr(run_base01_cpu, "run_in_process_group_bounded", run)
    slot = run_base01_cpu.collect_slot(
        SAMPLES[0], IRPath.MAKE_FX_ATEN, tmp_path
    )
    assert slot["observation"] is None
    assert slot["files"]["stdout.log"]
    summary = run_base01_cpu.summarize([slot])
    assert summary["f_planned_slots"] == 32
    assert summary["f_attempted_slots"] == 1
    assert summary["f_accounting_completed"] == 0
    assert summary["formal_qualified"] == 0


def test_output_directory_cannot_be_overwritten(tmp_path):
    output = tmp_path / SAMPLES[0].sample_id / IRPath.MAKE_FX_ATEN.value
    output.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        run_base01_cpu.collect_slot(SAMPLES[0], IRPath.MAKE_FX_ATEN, tmp_path)


def test_source_identity_covers_current_counter_and_test_harness():
    files = base01_records.source_identity()["files"]
    assert "src/solar/analysis/graph_accounting.py" in files
    assert "scripts/internal/solar/base01_samples.py" in files
    assert "tests/solar/test_base01_samples.py" in files
    assert "uv.lock" in files


def test_real_inputs_have_two_unrun_paths_and_exact_frozen_identities():
    records = base01_records.real_input_records()
    assert [record["id"] for record in records] == [
        f"R{i:02}" for i in range(1, 9)
    ]
    assert all(len(record["paths"]) == 2 for record in records)
    assert all(
        path["status"] == "not_run"
        for record in records
        for path in record["paths"]
    )


def test_cpu_guard_refuses_visible_gpu_device_nodes(tmp_path, monkeypatch):
    node = tmp_path / "kfd"
    node.touch()
    monkeypatch.setattr(base01_records, "GPU_NODES", (node,))
    with pytest.raises(RuntimeError, match="without GPU nodes"):
        base01_records.require_cpu_isolation()


@pytest.mark.parametrize(
    "variant", ["missing", "malformed"], ids=["missing-receipt", "invalid-json"]
)
def test_successful_exit_does_not_hide_invalid_receipt(
    tmp_path, monkeypatch, variant
):
    def run(command, **kwargs):
        if variant == "malformed":
            output = tmp_path / SAMPLES[0].sample_id / IRPath.MAKE_FX_ATEN.value
            (output / "observation.json").write_text("{broken")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(run_base01_cpu, "run_in_process_group_bounded", run)
    slot = run_base01_cpu.collect_slot(
        SAMPLES[0], IRPath.MAKE_FX_ATEN, tmp_path
    )
    assert slot["observation"] is None
    expected = (
        ReceiptFailure.MISSING
        if variant == "missing"
        else ReceiptFailure.INVALID
    )
    assert slot["receipt_failure"] is expected
    summary = run_base01_cpu.summarize([slot])
    assert summary["f_attempted_slots"] == 1
    assert summary["f_failures"] == {f"worker_{expected}": 1}

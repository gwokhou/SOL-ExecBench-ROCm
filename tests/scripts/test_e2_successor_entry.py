"""Check the E2 entry boundaries without drawing or using a holdout key."""

import subprocess
from pathlib import Path

import pytest

from scripts.internal.solar import (
    e2_successor_entry as entry,
    e2_successor_exposure as exposure,
)
from scripts.internal.solar.e2_successor_campaign import AttemptAllowance
from scripts.internal.solar.e2_successor_freeze import VerifiedSuccessorFreeze
from scripts.internal.solar.e2_successor_ledger import (
    Phase,
    SuccessorLedger,
    planned_sessions,
)
from sol_execbench.core.bench.clock_ownership import (
    ClockDevice,
    ClockOwnerRecord,
)
from sol_execbench.core.data.json_utils import atomic_write_json_value


def test_online_refuses_before_touching_the_freeze_without_authorization(
    tmp_path, monkeypatch, capsys
):
    def reject(*_args, **_kwargs):
        pytest.fail("authorization must precede freeze verification")

    monkeypatch.setattr(entry, "verify_frozen_successor", reject)
    monkeypatch.setattr(exposure.secrets, "token_bytes", reject)
    with pytest.raises(SystemExit) as raised:
        entry.main(
            [
                "online",
                "--repository",
                str(tmp_path),
                "--freeze",
                str(tmp_path / "freeze.json"),
                "--evidence-root",
                str(tmp_path),
            ]
        )
    assert raised.value.code == 2
    assert "requires explicit --open-holdout" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


def test_commitment_authorization_precedes_key_draw(tmp_path, monkeypatch):
    ledger = SuccessorLedger.create(tmp_path / "study", freeze_sha256="a" * 64)
    freeze = VerifiedSuccessorFreeze(
        path=tmp_path / "freeze.json",
        sha256="a" * 64,
        schedule_sha256="b" * 64,
        execution_commit="c" * 40,
        image_id="sha256:" + "d" * 64,
        device=ClockDevice(uuid="fixture", bdf="fixture"),
        activity=ledger.root,
    )

    def reject(*_args, **_kwargs):
        pytest.fail("unauthorized E2 setup must not draw a key")

    monkeypatch.setattr(exposure.secrets, "token_bytes", reject)
    monkeypatch.setattr(exposure, "schedule_sha256", reject)
    with pytest.raises(
        PermissionError, match="requires explicit authorization"
    ):
        exposure.commit_successor_holdout(ledger, freeze, authorized=False)
    assert {file.name for file in ledger.root.iterdir()} == {"roster.json"}


def test_heldout_worker_requires_its_own_commitment(tmp_path):
    stage = tmp_path / "study/online/session-0006/evidence"
    with pytest.raises(
        ValueError, match="lacks its committed exposure receipt"
    ):
        exposure.verify_worker_commitment(stage, "b" * 64)
    with pytest.raises(
        ValueError, match="outside the committed online session"
    ):
        exposure.verify_worker_commitment(Path("evidence"), "b" * 64)


def test_offline_timeout_retains_drained_evidence_and_original_verdict(
    tmp_path, monkeypatch
):
    ledger = SuccessorLedger.create(tmp_path / "study", freeze_sha256="a" * 64)
    item = next(planned_sessions(Phase.OFFLINE))
    source = ledger.root / Phase.ONLINE / item.directory / "evidence"
    source.mkdir(parents=True)
    atomic_write_json_value(
        source / "session.json", {"original_evaluator_verdict": "PASSED"}
    )
    destination = ledger.root / Phase.OFFLINE / item.directory / "evidence"

    def terminate(command, **kwargs):
        assert kwargs["timeout"] == 10
        assert command[1:6] == [
            "-B",
            "-P",
            "-m",
            entry._ENTRY_MODULE,
            "replay-one",
        ]
        destination.mkdir(parents=True)
        atomic_write_json_value(
            destination / "result.json",
            {
                "session_id": item.session.identity,
                "gpu_initialized": False,
                "oracle_cpu_seconds": 0.35,
                "stop_reason": "SystemExit: 143",
            },
        )
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(entry, "run_in_process_group_bounded", terminate)
    outcome = entry._run_offline_attempt(
        tmp_path,
        ledger,
        item,
        destination,
        AttemptAllowance(
            wall_seconds=10, oracle_cpu_seconds=100, output_bytes=1024
        ),
    )
    assert outcome.evidence == destination / "result.json"
    assert outcome.verdict == "PASSED"
    assert outcome.stop_reason == "SystemExit: 143"
    assert outcome.oracle_cpu_seconds == 0.35


def test_offline_accepts_only_the_frozen_owners_restoration(tmp_path):
    ledger = SuccessorLedger.create(tmp_path / "study", freeze_sha256="a" * 64)
    device = ClockDevice(uuid="fixture", bdf="fixture")
    owner = ClockOwnerRecord(
        lease_id="a" * 32,
        devices=(device,),
        owner_pid=123,
        started_at="2026-11-18T00:00:00Z",
        container_supervised=True,
    ).model_dump(mode="json")
    atomic_write_json_value(ledger.root / entry._OWNER_START, owner)
    closeout = ledger.root / "gpu-closeout.json"
    ended = {
        **owner,
        "state": "restored",
        "ended_at": "2026-11-18T05:59:59Z",
    }
    receipt = {
        "returncode": 0,
        "admission": {"state": "ready", "owner": None},
        "receipts": {f"{owner['lease_id']}.json": ended},
    }
    atomic_write_json_value(closeout, receipt)
    entry._verify_native_closeout(ledger, closeout)
    ended["ended_at"] = "2026-11-18T06:00:01Z"
    atomic_write_json_value(closeout, receipt)
    with pytest.raises(ValueError, match="not restored within"):
        entry._verify_native_closeout(ledger, closeout)

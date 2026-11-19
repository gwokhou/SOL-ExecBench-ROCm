"""Exercise complete denominators and once-only E2 phase receipts on CPU."""

import time
from pathlib import Path

import pytest

from scripts.internal.solar import e2_successor_campaign as campaign
from scripts.internal.solar.e2_successor_campaign import (
    AttemptAllowance,
    AttemptOutcome,
    bind_gpu_handback,
    execute_phase,
)
from scripts.internal.solar.e2_successor_ledger import (
    AttemptState,
    Phase,
    PlannedSession,
    SuccessorLedger,
    planned_sessions,
)
from sol_execbench.core.data.json_utils import atomic_write_json_value

_FREEZE = "a" * 64


def test_ledger_freezes_the_complete_denominator_without_materializing_inputs(
    tmp_path,
):
    ledger = SuccessorLedger.create(tmp_path / "study", freeze_sha256=_FREEZE)
    assert ledger.manifest["logical_units"] == 432
    assert ledger.manifest["correctness_sessions"] == 1296
    assert ledger.manifest["cost_sessions"] == 144
    assert ledger.manifest["offline_sessions"] == 72
    online = ledger.reconcile(Phase.ONLINE)
    assert len(online) == len({row["session_id"] for row in online}) == 1440
    assert all(row["state"] == AttemptState.NOT_REACHED for row in online)
    assert {p.name for p in ledger.root.iterdir()} == {"roster.json"}


def test_missing_final_receipt_preserves_unknown_and_prevents_reexecution(
    tmp_path,
):
    ledger = SuccessorLedger.create(tmp_path / "study", freeze_sha256=_FREEZE)
    first = next(planned_sessions(Phase.ONLINE))
    ledger.begin_phase(Phase.ONLINE)
    ledger.start(Phase.ONLINE, first)
    with pytest.raises(FileExistsError):
        ledger.start(Phase.ONLINE, first)
    first_row, *_ = ledger.reconcile(Phase.ONLINE)
    assert first_row["state"] == AttemptState.INTERRUPTED
    assert first_row["verdict"] is None
    with pytest.raises(FileExistsError):
        ledger.begin_phase(Phase.ONLINE)


def test_infrastructure_failure_stops_on_first_unfinished_attempt(
    tmp_path, monkeypatch
):
    ledger = SuccessorLedger.create(tmp_path / "study", freeze_sha256=_FREEZE)
    monkeypatch.setattr(
        campaign, "admit_evidence_writer", lambda *_a, **_k: None
    )
    attempted = []

    def run(
        item: PlannedSession, destination: Path, limit: AttemptAllowance
    ) -> AttemptOutcome:
        attempted.append(item.index)
        assert limit.wall_seconds > 0
        assert limit.output_bytes > 0
        if item.index == 1:
            raise TimeoutError("worker exceeded its time limit")
        destination.mkdir()
        evidence = destination / "session.json"
        evidence.write_text("observed_numeric_rejection")
        return AttemptOutcome(
            verdict="INCORRECT_NUMERICAL",
            evidence=evidence,
            oracle_cpu_seconds=0.1,
        )

    status = execute_phase(
        ledger,
        Phase.ONLINE,
        run,
        online_owner_started_monotonic=time.monotonic(),
    )
    assert status["state"] == "INCOMPLETE"
    assert status["planned_attempts"] == 1440
    assert status["completed_attempts"] == 1
    assert status["total_oracle_cpu_seconds"] == 0.1
    assert status["oracle_cpu_spend_complete"] is False
    assert attempted == [0, 1]
    first, second, third, *_ = ledger.reconcile(Phase.ONLINE)
    assert first["state"] == AttemptState.COMPLETED
    assert first["verdict"] == "INCORRECT_NUMERICAL"
    assert second["state"] == AttemptState.INCOMPLETE
    assert second["verdict"] is None
    assert third["state"] == AttemptState.NOT_REACHED

    path = ledger.root / "online/session-0000" / first["evidence_path"]
    path.write_text("changed")
    with pytest.raises(ValueError, match="bound attempt evidence changed"):
        ledger.reconcile(Phase.ONLINE)


def test_offline_phase_rejects_bad_handback_and_preserves_roster(tmp_path):
    ledger = SuccessorLedger.create(tmp_path / "study", freeze_sha256=_FREEZE)
    closeout = ledger.root / "gpu-closeout.json"
    atomic_write_json_value(
        closeout,
        {
            "clean": True,
            "gpu_process_count": 0,
            "locks": [
                {"level": "L1-ownership", "held": False},
                {"level": "L2-execution", "held": True},
            ],
            "clocks": {
                "gpu_data": [{"perf_level": "AMDSMI_DEV_PERF_LEVEL_AUTO"}]
            },
        },
    )
    with pytest.raises(ValueError, match="native GPU handback is incomplete"):
        bind_gpu_handback(ledger, closeout)
    assert len(ledger.reconcile(Phase.OFFLINE)) == 72
    assert not (ledger.root / "offline").exists()


def test_clean_gpu_handback_does_not_replace_completed_attempt_evidence(
    tmp_path,
):
    ledger = SuccessorLedger.create(tmp_path / "study", freeze_sha256=_FREEZE)
    ledger.begin_phase(Phase.ONLINE)
    atomic_write_json_value(
        ledger.root / "online/phase-result.json",
        {
            "state": "completed",
            "completed_attempts": 1440,
            "roster_sha256": ledger.roster_sha256,
            "total_oracle_cpu_seconds": 0.0,
        },
    )
    closeout = ledger.root / "gpu-closeout.json"
    atomic_write_json_value(
        closeout,
        {
            "clean": True,
            "gpu_process_count": 0,
            "locks": [
                {"level": "L1-ownership", "held": False},
                {"level": "L2-execution", "held": False},
            ],
            "clocks": {
                "gpu_data": [{"perf_level": "AMDSMI_DEV_PERF_LEVEL_AUTO"}]
            },
        },
    )
    bind_gpu_handback(ledger, closeout)

    def run(*_args):
        pytest.fail("offline replay started on unverified online evidence")

    with pytest.raises(
        ValueError, match="inconsistent with the attempt receipts"
    ):
        execute_phase(ledger, Phase.OFFLINE, run)
    assert not (ledger.root / "offline").exists()

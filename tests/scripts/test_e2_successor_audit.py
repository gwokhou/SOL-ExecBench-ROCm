"""Test evidence linkage and interrupted denominators without opening holdout."""

import numpy as np
import pytest

from scripts.internal.solar import e2_successor_audit as audit
from scripts.internal.solar.e2_offline_storage import (
    commit_diagnostic_bundle,
    encode_diagnostic_arrays,
)
from scripts.internal.solar.e2_successor_freeze import VerifiedSuccessorFreeze
from scripts.internal.solar.e2_successor_ledger import (
    AttemptState,
    Phase,
    SuccessorLedger,
    planned_sessions,
)
from sol_execbench.core.bench.clock_ownership import ClockDevice
from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.integrity import sha256_file


@pytest.fixture
def sealed_empty_study(tmp_path):
    freeze = VerifiedSuccessorFreeze(
        path=tmp_path / "frozen.json",
        sha256="a" * 64,
        schedule_sha256="b" * 64,
        execution_commit="c" * 40,
        image_id="sha256:" + "d" * 64,
        device=ClockDevice(uuid="fixture", bdf="fixture"),
        activity=tmp_path / "study",
    )
    return SuccessorLedger.create(
        freeze.activity, freeze_sha256=freeze.sha256
    ), freeze


def test_empty_audit_preserves_both_denominators(sealed_empty_study):
    ledger, freeze = sealed_empty_study
    result = audit.audit_evidence(ledger, freeze)
    assert result["state"] == "INCOMPLETE"
    assert len(result["online"]) == 1440
    assert len(result["offline"]) == 72
    assert not result["holdout_commitment_verified"]
    assert {file.name for file in ledger.root.iterdir()} == {"roster.json"}


def test_interrupted_attempt_retains_unknowns(sealed_empty_study):
    ledger, freeze = sealed_empty_study
    first = next(planned_sessions(Phase.ONLINE))
    atomic_write_json_value(
        ledger.root / audit.COMMITMENT_FILE,
        {
            "research_format": audit._COMMITMENT_KIND,
            "freeze_sha256": freeze.sha256,
            "schedule_sha256": freeze.schedule_sha256,
            "roster_sha256": ledger.roster_sha256,
            "key_sha256": "f" * 64,
            "created_unix": 1.0,
            "instances_materialized_at_commitment": 0,
            "performance_authority": False,
        },
    )
    ledger.begin_phase(Phase.ONLINE)
    ledger.start(Phase.ONLINE, first)
    result = audit.audit_evidence(ledger, freeze)
    interrupted, future, *_ = result["online"]
    assert interrupted["state"] == AttemptState.INTERRUPTED
    assert interrupted["verdict"] is None
    assert interrupted["audited_calls"] is None
    assert interrupted["replayed_input_hashes"] is None
    assert future["state"] == AttemptState.NOT_REACHED


def test_complete_phase_cannot_skip_its_attempt_receipts(sealed_empty_study):
    ledger, freeze = sealed_empty_study
    ledger.begin_phase(Phase.ONLINE)
    atomic_write_json_value(
        ledger.root / Phase.ONLINE / "phase-result.json",
        {
            "state": "completed",
            "phase": "online",
            "roster_sha256": ledger.roster_sha256,
            "planned_attempts": 1440,
            "completed_attempts": 1440,
            "performance_authority": False,
        },
    )
    with pytest.raises(ValueError, match="differs from the frozen ledger"):
        audit.audit_evidence(ledger, freeze)


def test_offline_audit_catches_changed_derived_archive(tmp_path):
    source = tmp_path / "session.json"
    raw = tmp_path / "judge-observations/call-000.npz"
    raw.parent.mkdir()
    raw.write_bytes(b"captured data")
    metadata = {"input_sha256": "d" * 64}
    capture = [
        {**metadata, "artifact": raw.name, "artifact_sha256": sha256_file(raw)}
    ]
    destination = tmp_path / "replay"
    destination.mkdir()
    encoded = encode_diagnostic_arrays(
        {
            "oracle_rounded": np.array([1], dtype=np.float32),
            "oracle_exact": np.array([1], dtype=np.float64),
        }
    )
    commit_diagnostic_bundle(
        destination,
        0,
        encoded,
        metadata,
        source_sha256=sha256_file(raw),
        remaining_bytes=1024**2,
    )
    audit._verify_offline_calls(destination, source, capture, 1)
    (destination / "call-000/diagnostic.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="archive identity changed"):
        audit._verify_offline_calls(destination, source, capture, 1)

"""Independently reconcile sealed E2 evidence without executing candidates."""

import argparse
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from scripts.internal.solar.e2_contracts import (
    EVIDENCE_BUDGET_BYTES,
    MAX_SESSION_OBSERVATIONS,
    ORACLE_BUDGET_SECONDS,
    Activity,
    base_seed,
)
from scripts.internal.solar.e2_evidence_budget import (
    admit_evidence_writer,
    retained_evidence,
)
from scripts.internal.solar.e2_input_replay import verify_session_inputs
from scripts.internal.solar.e2_offline_storage import (
    CALL_RECEIPT,
    DIAGNOSTIC_ARCHIVE,
    load_diagnostic_arrays,
)
from scripts.internal.solar.e2_postprocess import decode_observation
from scripts.internal.solar.e2_successor_campaign import (
    _HANDBACK,
    _PHASE_RESULT,
    _offline_oracle_spend,
)
from scripts.internal.solar.e2_successor_entry import (
    _NATIVE_CLOSEOUT,
    _require_device_free,
    _verify_native_closeout,
)
from scripts.internal.solar.e2_successor_exposure import (
    _CLOSED_KEY,
    _COMMITMENT_KIND,
    COMMITMENT_FILE,
)
from scripts.internal.solar.e2_successor_freeze import (
    VerifiedSuccessorFreeze,
    verify_frozen_successor,
)
from scripts.internal.solar.e2_successor_ledger import (
    AttemptState,
    Phase,
    PlannedSession,
    SuccessorLedger,
    planned_sessions,
)
from sol_execbench.core.bench.judge.contracts import JudgeInstance
from sol_execbench.core.bench.judge.storage import tensor_identity
from sol_execbench.core.data.json_utils import (
    atomic_write_json_value,
    canonical_json_bytes,
    load_json_value,
)
from sol_execbench.core.integrity import sha256_bytes, sha256_file
from sol_execbench.core.process.environment import ENV_SOL_EXECBENCH_SANDBOXED

_OBSERVATIONS = Path("judge-observations/observations.json")
_AUDIT_NAME = "independent-evidence-audit.json"
_FULL_PATH_CALLS = {
    Activity.CORRECTNESS: (11, 12),
    Activity.COST: (MAX_SESSION_OBSERVATIONS, 371),
}


def _phase_close(
    ledger: SuccessorLedger, phase: Phase, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    folder = ledger.root / phase
    path = folder / _PHASE_RESULT
    counts = dict(Counter(row["state"] for row in rows))
    if not path.exists():
        return {
            "state": AttemptState.INTERRUPTED
            if folder.exists()
            else AttemptState.NOT_REACHED,
            "counts": counts,
            "receipt_sha256": None,
        }
    close = load_json_value(path)
    if (
        close["roster_sha256"] != ledger.roster_sha256
        or close["phase"] != phase
        or close["planned_attempts"] != len(rows)
        or close["completed_attempts"] != counts.get(AttemptState.COMPLETED, 0)
        or close["state"] not in {"completed", "INCOMPLETE"}
        or close["performance_authority"] is not False
    ):
        raise ValueError(f"E2 {phase} close differs from the frozen ledger")
    if close["state"] == AttemptState.COMPLETED:
        seconds = close["total_oracle_cpu_seconds"]
        if (
            counts != {AttemptState.COMPLETED: len(rows)}
            or close["stop_reason"] is not None
            or close["oracle_cpu_spend_complete"] is not True
            or not math.isfinite(seconds)
            or not 0 <= seconds <= ORACLE_BUDGET_SECONDS
        ):
            raise ValueError(f"E2 {phase} completion lacks complete evidence")
    return {**close, "counts": counts, "receipt_sha256": sha256_file(path)}


def _closed_key(
    ledger: SuccessorLedger, freeze: VerifiedSuccessorFreeze, state: str
) -> bytes | None:
    path = ledger.root / COMMITMENT_FILE
    key_path = ledger.root / _CLOSED_KEY
    if not path.exists():
        if key_path.exists() or state != AttemptState.NOT_REACHED:
            raise ValueError("E2 online work lacks its preceding commitment")
        return None
    commitment = load_json_value(path)
    expected = {
        "research_format": _COMMITMENT_KIND,
        "freeze_sha256": freeze.sha256,
        "schedule_sha256": freeze.schedule_sha256,
        "roster_sha256": ledger.roster_sha256,
        "instances_materialized_at_commitment": 0,
        "performance_authority": False,
    }
    if any(commitment[k] != v for k, v in expected.items()):
        raise ValueError("E2 holdout commitment differs from the sealed roster")
    if state != AttemptState.NOT_REACHED:
        start = load_json_value(ledger.root / Phase.ONLINE / "started.json")
        if (
            start["roster_sha256"] != ledger.roster_sha256
            or not 0 < commitment["created_unix"] <= start["started_unix"]
        ):
            raise ValueError(
                "E2 commitment did not precede the bound online phase"
            )
    if not key_path.exists():
        if state == AttemptState.COMPLETED:
            raise ValueError("E2 complete online activity lacks its closed key")
        return None
    if state not in {AttemptState.COMPLETED, "INCOMPLETE"}:
        raise ValueError("E2 key appeared before the terminal online receipt")
    with key_path.open("rb") as handle:
        key = handle.read(33)
    if len(key) != 32 or sha256_bytes(key) != commitment["key_sha256"]:
        raise ValueError("E2 closed key does not match its earlier commitment")
    return key


def _session_evidence(
    ledger: SuccessorLedger,
    phase: Phase,
    item: PlannedSession,
    row: dict[str, Any],
) -> Path | None:
    path = row.get("evidence_path")
    if path is None:
        if row["state"] == AttemptState.COMPLETED:
            raise ValueError("E2 completed attempt lacks its bound evidence")
        return None
    return ledger.root / phase / item.directory / path


def _captured(
    receipt: Path, item: PlannedSession, row: dict[str, Any]
) -> dict[str, Any] | None:
    closed = load_json_value(receipt)
    if (
        any(closed[k] != v for k, v in item.session.metadata().items())
        or closed["original_evaluator_verdict"] != row["verdict"]
    ):
        raise ValueError(
            "E2 online evidence has a mismatched source or verdict"
        )
    capture = closed.get("observations")
    if capture is None:
        if row["state"] == AttemptState.COMPLETED:
            raise ValueError("E2 completed online evidence has no actual calls")
        return None
    if (
        capture != load_json_value(receipt.parent / _OBSERVATIONS)
        or capture["publication_after_candidate_exit"] is not True
        or len(capture["observations"]) > MAX_SESSION_OBSERVATIONS
    ):
        raise ValueError("E2 capture source is inconsistent")
    if row["state"] == AttemptState.COMPLETED and (
        capture["stop_reason"] or capture["pending_correctness"]
    ):
        raise ValueError("E2 completed online evidence has unfinished calls")
    if row["state"] == AttemptState.COMPLETED and row["verdict"] == "PASSED":
        counts = (len(capture["observations"]), capture["reference_calls"])
        if counts != _FULL_PATH_CALLS[item.session.activity]:
            raise ValueError("E2 accepted verdict lacks the full call path")
    return capture


def _verify_raw_calls(
    receipt: Path, capture: dict[str, Any], seen: set[str]
) -> int:
    count = 0
    for idx, observation in enumerate(capture["observations"]):
        name = f"call-{idx:03d}.npz"
        archive = receipt.parent / _OBSERVATIONS.parent / name
        if (
            observation["artifact"] != name
            or sha256_file(archive) != observation["artifact_sha256"]
        ):
            raise ValueError(
                "E2 raw call archive no longer matches its receipt"
            )
        tensors = decode_observation(archive, observation)
        identity = observation["input_sha256"]
        if (
            tensor_identity(tensors.inputs.values) != identity
            or tensor_identity((tensors.reference,))
            != observation["reference_sha256"]
            or tensor_identity((tensors.actual,))
            != observation["candidate_output_sha256"]
            or identity in seen
        ):
            raise ValueError(
                "E2 captured tensor identity or input uniqueness failed"
            )
        seen.add(identity)
        count += 1
    return count


def _audit_online(
    ledger: SuccessorLedger, rows: list[dict[str, Any]], key: bytes | None
) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result = []
    for item, row in zip(planned_sessions(Phase.ONLINE), rows, strict=True):
        receipt = _session_evidence(ledger, Phase.ONLINE, item, row)
        raw_count = replayed = None
        if receipt is not None:
            capture = _captured(receipt, item, row)
            if capture is not None:
                raw_count = _verify_raw_calls(receipt, capture, seen)
                case = item.session.case
                if (
                    case.instance is not JudgeInstance.HOLDOUT
                    or key is not None
                ):
                    replayed = len(
                        verify_session_inputs(
                            receipt.parent,
                            item.session,
                            capture,
                            private_seed=base_seed(case, key or b""),
                        )
                    )
        result.append(
            {
                **row,
                "audited_calls": raw_count,
                "replayed_input_hashes": replayed,
            }
        )
    return result


def _verify_offline_calls(
    destination: Path, source: Path, available: list[dict[str, Any]], count: int
) -> None:
    for idx, observation in enumerate(available[:count]):
        call = destination / f"call-{idx:03d}"
        receipt = load_json_value(call / CALL_RECEIPT)
        if (
            receipt["source_sha256"] != observation["artifact_sha256"]
            or receipt["input_sha256"] != observation["input_sha256"]
        ):
            raise ValueError(
                "E2 offline diagnostic differs from its actual call"
            )
        raw = source.parent / _OBSERVATIONS.parent / observation["artifact"]
        load_diagnostic_arrays(raw, call / DIAGNOSTIC_ARCHIVE, receipt)


def _audit_offline(
    ledger: SuccessorLedger,
    rows: list[dict[str, Any]],
    online: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for item, row in zip(planned_sessions(Phase.OFFLINE), rows, strict=True):
        receipt = _session_evidence(ledger, Phase.OFFLINE, item, row)
        count = None
        if receipt is not None:
            source = _session_evidence(
                ledger, Phase.ONLINE, item, online[item.index]
            )
            if source is None:
                raise ValueError(
                    "E2 offline work lacks its bound online source"
                )
            manifest = source.parent / _OBSERVATIONS
            available = load_json_value(manifest)["observations"]
            closed = load_json_value(receipt)
            count = closed["committed"]
            if (
                closed["gpu_initialized"] is not False
                or closed["session_id"] != item.session.identity
                or row["verdict"] != online[item.index]["verdict"]
                or closed["source_receipt_sha256"] != sha256_file(manifest)
                or len(available) != closed["available_observations"]
                or not 0 <= count <= len(available)
            ):
                raise ValueError(
                    "E2 offline receipt lacks a stable source binding"
                )
            if row["state"] == AttemptState.COMPLETED and (
                count != len(available)
                or closed["stop_reason"]
                or any(receipt.parent.glob(".pending-*"))
            ):
                raise ValueError(
                    "E2 completed offline receipt contains unfinished work"
                )
            _verify_offline_calls(receipt.parent, source, available, count)
        result.append({**row, "audited_diagnostics": count})
    return result


def audit_evidence(
    ledger: SuccessorLedger, freeze: VerifiedSuccessorFreeze
) -> dict[str, Any]:
    """Audit every reachable raw artifact while retaining the exact denominators."""
    online, offline = (
        ledger.reconcile(Phase.ONLINE),
        ledger.reconcile(Phase.OFFLINE),
    )
    online_close = _phase_close(ledger, Phase.ONLINE, online)
    offline_close = _phase_close(ledger, Phase.OFFLINE, offline)
    key = _closed_key(ledger, freeze, online_close["state"])
    complete = (
        online_close["state"]
        == offline_close["state"]
        == AttemptState.COMPLETED
    )
    handback_sha = None
    if offline_close["state"] != AttemptState.NOT_REACHED:
        closeout = ledger.root / _NATIVE_CLOSEOUT
        _verify_native_closeout(ledger, closeout)
        online_cpu = _offline_oracle_spend(ledger)
        if complete and offline_close["total_oracle_cpu_seconds"] < online_cpu:
            raise ValueError(
                "E2 combined oracle budget omitted the online spend"
            )
        handback_sha = sha256_file(ledger.root / _HANDBACK)
    online_audit = _audit_online(ledger, online, key)
    offline_audit = _audit_offline(ledger, offline, online)
    usage = retained_evidence(ledger.root)
    if complete and (
        any(
            row["audited_calls"] != row["replayed_input_hashes"]
            for row in online_audit
        )
        or max(usage.allocated_bytes, usage.logical_bytes)
        > EVIDENCE_BUDGET_BYTES
        or torch.cuda.is_initialized()
    ):
        raise ValueError(
            "E2 completion lacks input replay or violates the budget envelope"
        )
    return {
        "research_format": "e2-successor-evidence-audit-v1",
        "state": "completed" if complete else "INCOMPLETE",
        "freeze_sha256": freeze.sha256,
        "roster_sha256": ledger.roster_sha256,
        "native_handback_sha256": handback_sha,
        "holdout_commitment_verified": key is not None,
        "online_phase": online_close,
        "offline_phase": offline_close,
        "online": online_audit,
        "offline": offline_audit,
        "retained_allocated_bytes": usage.allocated_bytes,
        "retained_logical_bytes": usage.logical_bytes,
        "gpu_initialized": torch.cuda.is_initialized(),
        "performance_authority": False,
    }


def main() -> None:
    """Write a bounded audit receipt in the frozen, device-free runtime."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--freeze", required=True, type=Path)
    args = parser.parse_args()
    if (
        not sys.flags.safe_path
        or os.environ.get(ENV_SOL_EXECBENCH_SANDBOXED) != "1"
    ):
        parser.error("E2 audit requires the hardened container and Python -P")
    _require_device_free()
    torch.set_num_threads(1)
    freeze = verify_frozen_successor(
        args.freeze.resolve(strict=True),
        repository=args.repository,
        capsule=Path(__file__).resolve().parents[3],
        evidence=args.evidence_root,
    )
    ledger = SuccessorLedger(freeze.activity, freeze_sha256=freeze.sha256)
    output = ledger.root / _AUDIT_NAME
    if output.exists():
        raise FileExistsError(output)
    result = audit_evidence(ledger, freeze)
    admit_evidence_writer(
        ledger.root,
        writer_limit_bytes=len(canonical_json_bytes(result)) + 1024**2,
    )
    atomic_write_json_value(output, result)
    print(result["state"])
    sys.exit(int(result["state"] != "completed"))


if __name__ == "__main__":
    main()

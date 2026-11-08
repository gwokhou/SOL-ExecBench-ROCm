"""One-shot E2 activity with precommitted exposure and complete denominators."""

import argparse
import json
import os
import secrets
import signal
import threading
import time
from pathlib import Path
from typing import Any

from scripts.internal.solar.e2_contracts import (
    EVIDENCE_BUDGET_BYTES,
    ORACLE_BUDGET_SECONDS,
    Mode,
    base_seed,
    schedule,
)
from scripts.internal.solar.e2_postprocess import diagnose_incumbent
from scripts.internal.solar.e2_runner import run_session
from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.integrity import sha256_bytes, sha256_file
from sol_execbench.core.process.resource_guard import evidence_size
from sol_execbench.core.process.termination import cleanup_on_termination
from sol_execbench.core.scoring.release_environment import (
    verify_release_source_state,
)
from solar.verification.execution import semantic_runtime_identity


def verify_freeze(repository: Path, freeze: dict[str, Any]) -> None:
    """Reject drift before creating a key or touching the held-out denominator."""
    verify_release_source_state(
        repository, expected_revision=freeze["execution_commit"]
    )
    for raw_path, digest in freeze["bound_files"].items():
        if sha256_file(Path(raw_path)) != digest:
            raise ValueError(f"E2 frozen identity changed: {raw_path}")
    if semantic_runtime_identity() != freeze["container_semantic_runtime"]:
        raise ValueError("E2 semantic runtime changed")
    if os.environ.get("SOL_EXECBENCH_CONTAINER_IMAGE_ID") != freeze["image_id"]:
        raise ValueError("E2 container image changed")
    prerequisite = json.loads(Path(freeze["prerequisite_result"]).read_text())
    if (
        prerequisite["status"] != "passed"
        or prerequisite["execution_commit"] != freeze["execution_commit"]
    ):
        raise ValueError("E2 prerequisite B2 identity is not qualified")


def _watch_budget(output: Path, deadline: float, stop: threading.Event) -> None:
    while not stop.wait(1):
        reason = None
        if time.time() >= deadline:
            reason = "gpu_activity_budget_exhausted"
        elif (
            evidence_size(output, EVIDENCE_BUDGET_BYTES)
            >= EVIDENCE_BUDGET_BYTES
        ):
            reason = "evidence_budget_exhausted"
        if reason:
            atomic_write_json_value(
                output / "budget-failure.json", {"reason": reason}
            )
            os.kill(os.getpid(), signal.SIGTERM)
            return


def _persist(
    output: Path,
    rows: list[dict[str, Any]],
    *,
    stop_reason: str | None,
    oracle_seconds: float,
) -> None:
    atomic_write_json_value(
        output / "ledger.json",
        {
            "research_format": "E2-exposure-ledger-v1",
            "performance_authority": False,
            "logical_units": 432,
            "correctness_sessions": 1296,
            "cost_sessions": 144,
            "stop_reason": stop_reason,
            "oracle_cpu_seconds": oracle_seconds,
            "status": "INCOMPLETE"
            if stop_reason or any(r["state"] != "completed" for r in rows)
            else "completed",
            "sessions": rows,
        },
    )


def _run_schedule(
    repository: Path,
    output: Path,
    key: bytes,
    deadline: float,
    rows: list[dict[str, Any]],
) -> tuple[str | None, float]:
    oracle_seconds = 0.0
    stop_reason = None
    for index, session in enumerate(schedule()):
        if time.time() >= deadline or oracle_seconds >= ORACLE_BUDGET_SECONDS:
            stop_reason = "activity_budget_exhausted"
            break
        row = rows[index]
        row["state"] = "running"
        row["started_unix"] = time.time()
        _persist(output, rows, stop_reason=None, oracle_seconds=oracle_seconds)
        print(
            json.dumps(
                {
                    "session_index": index,
                    "session_id": session.identity,
                    "phase": "start",
                }
            ),
            flush=True,
        )
        session_output = output / f"session-{index:04d}"
        try:
            result = run_session(
                repository,
                session_output,
                session,
                private_seed=base_seed(session.case, key),
                nonce=secrets.token_hex(32),
                remaining_seconds=deadline - time.time(),
            )
            row.update(
                state="completed",
                artifact=str(session_output.relative_to(output)),
                verdict=result["original_evaluator_verdict"],
            )
            observations = result.get("observations", {}).get(
                "observations", []
            )
            stop_reason = result["stop_reason"]
            if stop_reason is None and session.mode is Mode.INCUMBENT:
                offline = diagnose_incumbent(
                    repository, session_output, session
                )
                observations = offline["observations"]
                stop_reason = offline["stop_reason"]
            oracle_seconds += sum(
                item.get("oracle_cpu_seconds", 0.0) for item in observations
            )
            row["stop_reason"] = stop_reason
            if stop_reason:
                row["state"] = "incomplete"
        except BaseException as exc:  # noqa: BLE001 -- terminal ledger, no retry or recovery
            stop_reason = f"{type(exc).__name__}: {exc}"
            row.update(state="incomplete", stop_reason=stop_reason)
        _persist(
            output, rows, stop_reason=stop_reason, oracle_seconds=oracle_seconds
        )
        if stop_reason:
            break
    return stop_reason, oracle_seconds


def execute(repository: Path, output: Path, freeze_path: Path) -> None:
    """Commit once, run once, and retain every stopped or unreached session."""
    freeze = json.loads(freeze_path.read_text())
    verify_freeze(repository, freeze)
    if time.time() >= freeze["activity_deadline_unix"]:
        raise TimeoutError("E2 deadline expired before exposure")
    output.mkdir(parents=True, exist_ok=False)
    rows = [
        {**session.metadata(), "state": "not_reached", "verdict": "not_reached"}
        for session in schedule()
    ]
    _persist(output, rows, stop_reason="not_started", oracle_seconds=0.0)
    key = secrets.token_bytes(32)
    atomic_write_json_value(
        output / "holdout-commitment.json",
        {
            "sha256": sha256_bytes(key),
            "created_unix": time.time(),
            "freeze_sha256": sha256_file(freeze_path),
            "instances_materialized_at_commitment": 0,
        },
    )
    stop = threading.Event()
    watcher = threading.Thread(
        target=_watch_budget,
        args=(output, freeze["activity_deadline_unix"], stop),
        daemon=True,
    )
    watcher.start()
    stop_reason = "interrupted_before_first_session"
    oracle_seconds = 0.0
    try:
        with cleanup_on_termination():
            stop_reason, oracle_seconds = _run_schedule(
                repository, output, key, freeze["activity_deadline_unix"], rows
            )
    finally:
        stop.set()
        watcher.join(timeout=2)
        # No candidate process remains; permit replay of the prior commitment.
        key_path = output / "holdout-key-after-close.bin"
        key_path.write_bytes(key)
        key_path.chmod(0o600)
        _persist(
            output, rows, stop_reason=stop_reason, oracle_seconds=oracle_seconds
        )


def main() -> None:
    """Require an explicit frozen execution manifest and a fresh evidence root."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--freeze", required=True, type=Path)
    args = parser.parse_args()
    execute(args.repository, args.output, args.freeze)


if __name__ == "__main__":
    main()

"""Bounded development-only E2 preflight inside the fixed ROCm image."""

import argparse
import errno
import json
import time
from pathlib import Path
from typing import Any

import torch
from pilot_capture import INTERRUPT_AFTER, QUOTA_FAULT
from pilot_runner import (
    SCRATCH,
    SCRATCH_BYTES,
    bound_mount,
    run_session,
    scratch_usage,
)

from scripts.internal.solar.e2_contracts import (
    Activity,
    Mode,
    Session,
    schedule,
)
from scripts.internal.solar.e2_postprocess import diagnose_incumbent
from sol_execbench.core.bench.judge.contracts import (
    JudgeInstance,
    JudgeStrategy,
)
from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.process.resource_guard import evidence_size

REPO = Path("/sol-execbench")
OUTPUT = Path("/outputs")
CORRECTNESS = (("gemm", "G0", 1), ("sum", "R0", 2), ("layernorm", "L3", 1))
COST = (("gemm", "G0", 0), ("sum", "R0", 0), ("layernorm", "L3", 0))
BLOCK_SIZE = 4 * 1024**2


def select(
    activity: Activity, coordinates: tuple[str, str, int]
) -> list[Session]:
    """Select fixed pilot coordinates from the E0 schedule."""
    task, family, workload_index = coordinates
    return [
        session
        for session in schedule()
        if session.activity is activity
        and session.case.task == task
        and session.case.family == family
        and session.case.workload_index == workload_index
        and session.case.instance is JudgeInstance.DEVELOPMENT_0
        and session.repeat == 0
        and session.strategy is JudgeStrategy.REFERENCE
    ]


def capacity_check() -> dict[str, Any]:
    """Prove the enforced quota with a real kernel write refusal."""
    bound_mount()
    path = SCRATCH / "quota-check.bin"
    block = bytes(BLOCK_SIZE)
    written = 0
    try:
        with path.open("xb", buffering=0) as handle:
            while written <= SCRATCH_BYTES:
                try:
                    count = handle.write(block)
                except OSError as exc:
                    if exc.errno != errno.ENOSPC:
                        raise
                    break
                if not count:
                    raise RuntimeError("scratch fixture write did not advance")
                written += count
        if written != SCRATCH_BYTES or scratch_usage()[0] != SCRATCH_BYTES:
            raise RuntimeError("kernel did not enforce exact scratch quota")
    finally:
        path.unlink(missing_ok=True)
    bound_mount()
    return {
        "status": "passed",
        "limit_bytes": SCRATCH_BYTES,
        "bytes_before_enospc": written,
        "bytes_after_cleanup": scratch_usage()[0],
    }


def run_fault_checks(output: Path) -> None:
    """Exercise graceful and abrupt faults through the CPU evaluator."""
    summary = {"scratch_quota": capacity_check(), "interruptions": []}
    (case,) = select(Activity.CORRECTNESS, ("gemm", "G0", 0))
    for fault in ("term", "kill", QUOTA_FAULT):
        root = output / f"fault-{fault}"
        result = run_session(REPO, root, case, device="cpu", fault=fault)
        captured = json.loads((root / "capture-progress.json").read_text())
        persisted = len(result.get("observations", {}).get("observations", []))
        expected_raw = 0 if fault == "kill" else INTERRUPT_AFTER
        if (
            captured["captured_prefix"] != INTERRUPT_AFTER
            or persisted != expected_raw
        ):
            raise RuntimeError(
                f"fault receipt mismatch: {fault}, {captured}, {persisted}"
            )
        if (
            result["original_evaluator_verdict"] != "not_reached"
            or (root / "trace.json").exists()
        ):
            raise RuntimeError("interruption manufactured a final trace")
        if (
            fault == QUOTA_FAULT
            and result["stop_reason"] != "pilot_capture_quota_exhausted"
        ):
            raise RuntimeError("capture quota stop classification was lost")
        summary["interruptions"].append(
            {
                "fault": fault,
                "captured_prefix_count": INTERRUPT_AFTER,
                "available_verified_raw_calls": persisted,
                "raw_calls_lost": INTERRUPT_AFTER - persisted,
                "remaining_calls": "unknown",
                "final_verdict": None,
                "stop_reason": result["stop_reason"],
                "exit_code": result["returncode"],
            }
        )
    summary["gpu_initialized"] = torch.cuda.is_initialized()
    if summary["gpu_initialized"]:
        raise RuntimeError("device-free fixture initialized torch GPU")
    atomic_write_json_value(
        output / "fault-checks.json", {"status": "passed", **summary}
    )


def run_measurements(output: Path) -> None:
    """Run all declared development slots and preserve failed rows."""
    proof = json.loads((OUTPUT / "cpu/fault-checks.json").read_text())
    if proof["status"] != "passed":
        raise RuntimeError(
            "bounded device preflight requires passing CPU fault checks"
        )
    sessions = [
        session
        for activity, coordinates in (
            (Activity.CORRECTNESS, CORRECTNESS),
            (Activity.COST, COST),
        )
        for coordinate in coordinates
        for session in select(activity, coordinate)
    ]
    if len(sessions) != 9:
        raise RuntimeError("development pilot denominator changed")
    rows: list[dict[str, Any]] = [
        {
            **session.metadata(),
            "state": "not_reached",
            "directory": f"session-{index:02d}",
        }
        for index, session in enumerate(sessions)
    ]
    ledger = output / "ledger.json"
    atomic_write_json_value(ledger, rows)
    for row, session in zip(rows, sessions, strict=True):
        row["state"] = "started"
        atomic_write_json_value(ledger, rows)
        print(f"Starting {session.identity}", flush=True)
        try:
            directory = output / row["directory"]
            result = run_session(REPO, directory, session, device="cuda:0")
            row.update(
                state="incomplete" if result["stop_reason"] else "completed",
                verdict=result["original_evaluator_verdict"],
                stop_reason=result["stop_reason"],
                retained_bytes=evidence_size(directory, 2 * 1024**3),
                wall_seconds=result["session_wall_seconds"],
                scratch=result["scratch"],
                candidate_observations=len(
                    result.get("observations", {}).get("observations", [])
                ),
                reference_calls=result.get("observations", {}).get(
                    "reference_calls"
                ),
            )
        except BaseException as exc:
            row.update(state="incomplete", stop_reason=type(exc).__name__)
            raise
        finally:
            atomic_write_json_value(ledger, rows)
        print(json.dumps(row), flush=True)


def offline(index: int) -> None:
    """Measure deferred oracle work outside the GPU session."""
    output = OUTPUT / "gpu"
    rows = json.loads((output / "ledger.json").read_text())
    row = rows[index]
    if row["mode"] != Mode.INCUMBENT or row["state"] != "completed":
        raise ValueError(
            "offline replay requires a completed incumbent pilot session"
        )
    (session,) = (
        case for case in schedule() if case.identity == row["session_id"]
    )
    session.case.require_development()
    directory = output / row["directory"]
    started = time.monotonic()
    outcome = diagnose_incumbent(REPO, directory, session)
    atomic_write_json_value(
        directory / "offline-measurement.json",
        {
            "wall_seconds": time.monotonic() - started,
            "observations": len(outcome["observations"]),
            "stop_reason": outcome["stop_reason"],
            "oracle_cpu_seconds": sum(
                o.get("oracle_cpu_seconds", 0) for o in outcome["observations"]
            ),
            "diagnosed_bytes": sum(
                p.stat().st_size
                for p in directory.glob("judge-observations/*.diagnosed.npz")
            ),
            "gpu_initialized": torch.cuda.is_initialized(),
        },
    )


def main() -> None:
    """Parse the pilot phase and require a fresh evidence directory."""
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("mode", choices=("cpu", "gpu", "offline"))
    cli.add_argument("--index", type=int)
    args = cli.parse_args()
    if args.mode == "offline":
        if args.index is None:
            cli.error("offline requires --index")
        offline(args.index)
        return
    root = OUTPUT / args.mode
    root.mkdir(exist_ok=False)
    if args.mode == "cpu":
        run_fault_checks(root)
    else:
        run_measurements(root)


if __name__ == "__main__":
    main()

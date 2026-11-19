"""Run the frozen E2 successor under the native owner and then without a GPU."""

import argparse
import os
import sys
import time
from collections.abc import Sequence
from datetime import datetime
from functools import partial
from pathlib import Path
from subprocess import TimeoutExpired
from typing import Any

from scripts.internal.solar.e2_contracts import (
    EVIDENCE_BUDGET_BYTES,
    GPU_BUDGET_SECONDS,
)
from scripts.internal.solar.e2_successor_campaign import (
    ONLINE_CLOSEOUT_RESERVE_SECONDS,
    AttemptAllowance,
    AttemptOutcome,
    bind_gpu_handback,
    execute_phase,
)
from scripts.internal.solar.e2_successor_exposure import (
    commit_successor_holdout,
    protect_observation_process,
)
from scripts.internal.solar.e2_successor_freeze import (
    VerifiedSuccessorFreeze,
    verify_frozen_successor,
)
from scripts.internal.solar.e2_successor_ledger import (
    Phase,
    PlannedSession,
    SuccessorLedger,
    planned_sessions,
)
from scripts.internal.solar.e2_successor_offline import (
    OfflineReplay,
    replay_incumbent,
)
from scripts.internal.solar.e2_successor_runner import (
    require_empty_scratch,
    run_staged_attempt,
)
from sol_execbench.core.bench.clock_ownership import (
    ENV_CLOCK_OWNER,
    ClockOwnerRecord,
    ClockOwnerState,
    verify_clock_owner,
)
from sol_execbench.core.bench.gpu_execution import (
    native_gpu_execution,
    resolve_gpu_device,
)
from sol_execbench.core.bench.gpu_lock import require_supervised_gpu_execution
from sol_execbench.core.bench.gpu_lock_paths import configured_gpu_locks
from sol_execbench.core.data.json_utils import (
    atomic_write_json_value,
    load_json_value,
)
from sol_execbench.core.process.environment import ENV_SOL_EXECBENCH_SANDBOXED
from sol_execbench.core.process.subprocesses import run_in_process_group_bounded

_ENTRY_MODULE = "scripts.internal.solar.e2_successor_entry"
_NATIVE_CLOSEOUT = "gpu-closeout.json"
_OWNER_START = "online-owner.json"
_ENTRY_CAPTURE_BYTES = 8192


def _clock_time(value: str) -> float:
    return datetime.fromisoformat(value).timestamp()


def _live_owner(
    freeze: VerifiedSuccessorFreeze,
) -> tuple[ClockOwnerRecord, float]:
    require_supervised_gpu_execution()
    device = resolve_gpu_device()
    if device != freeze.device:
        raise ValueError("native GPU differs from the frozen E2 device")
    record = verify_clock_owner(
        configured_gpu_locks().directory, os.environ[ENV_CLOCK_OWNER], (device,)
    )
    elapsed = time.time() - _clock_time(record.started_at)
    if (
        record.container_supervised is not True
        or not 0
        <= elapsed
        < GPU_BUDGET_SECONDS - ONLINE_CLOSEOUT_RESERVE_SECONDS
    ):
        raise ValueError("native owner has no E2 execution window")
    return record, time.monotonic() - elapsed


def _run_online(
    repository: Path, freeze: VerifiedSuccessorFreeze, *, authorized: bool
) -> dict[str, Any]:
    if authorized is not True:
        raise PermissionError(
            "E2 online execution requires explicit holdout authorization"
        )
    if freeze.activity.exists():
        raise FileExistsError(freeze.activity)
    if not configured_gpu_locks().host_managed:
        raise ValueError("E2 online execution requires the native host owner")
    require_empty_scratch()
    disk = os.statvfs(freeze.activity.parent)
    if disk.f_bavail * disk.f_frsize < EVIDENCE_BUDGET_BYTES:
        raise OSError("E2 activity filesystem lacks the frozen evidence budget")
    with native_gpu_execution(manage_clocks=True):
        owner, started = _live_owner(freeze)
        protect_observation_process()
        ledger = SuccessorLedger.create(
            freeze.activity, freeze_sha256=freeze.sha256
        )
        atomic_write_json_value(
            ledger.root / _OWNER_START, owner.model_dump(mode="json")
        )
        holdout = commit_successor_holdout(
            ledger, freeze, authorized=authorized
        )

        def run(
            item: PlannedSession, output: Path, allowance: AttemptAllowance
        ) -> AttemptOutcome:
            return run_staged_attempt(
                repository,
                item,
                output,
                allowance,
                seed=holdout.seed(item),
                holdout_commitment_sha256=holdout.commitment_sha256,
            )

        result = execute_phase(
            ledger, Phase.ONLINE, run, online_owner_started_monotonic=started
        )
        holdout.reveal_after_online_close()
        return result


def _verify_native_closeout(ledger: SuccessorLedger, closeout: Path) -> None:
    started = ClockOwnerRecord.model_validate_json(
        (ledger.root / _OWNER_START).read_text()
    )
    native = load_json_value(closeout)
    restored = ClockOwnerRecord.model_validate(
        native["receipts"][f"{started.lease_id}.json"]
    )
    if (
        native["returncode"] != 0
        or native["admission"]["state"] != "ready"
        or native["admission"]["owner"] is not None
        or restored.lease_id != started.lease_id
        or restored.started_at != started.started_at
        or restored.devices != started.devices
        or restored.state is not ClockOwnerState.RESTORED
        or restored.release_blocker is not None
        or restored.ended_at is None
        or not 0
        <= _clock_time(restored.ended_at) - _clock_time(started.started_at)
        <= GPU_BUDGET_SECONDS
    ):
        raise ValueError(
            "E2 owner is not restored within the frozen GPU window"
        )


def _require_device_free() -> None:
    if Path("/dev/kfd").exists() or any(Path("/dev/dri").glob("renderD*")):
        raise ValueError(
            "E2 offline execution requires a device-free container"
        )


def _replay_child(args: argparse.Namespace) -> int:
    _require_device_free()
    item = next(
        (i for i in planned_sessions(Phase.OFFLINE) if i.index == args.ordinal),
        None,
    )
    if item is None:
        raise ValueError("E2 offline ordinal is outside the frozen cost roster")
    result = replay_incumbent(
        OfflineReplay(
            repository=args.repository,
            source=args.activity / Phase.ONLINE / item.directory / "evidence",
            output=args.activity / Phase.OFFLINE / item.directory / "evidence",
            session=item.session,
            output_allowance_bytes=args.output_bytes,
            oracle_cpu_allowance_seconds=args.oracle_cpu_seconds,
        )
    )
    return int(result["stop_reason"] is not None)


def _run_offline_attempt(
    repository: Path,
    ledger: SuccessorLedger,
    item: PlannedSession,
    output: Path,
    allowance: AttemptAllowance,
) -> AttemptOutcome:
    command = [
        sys.executable,
        "-B",
        "-P",
        "-m",
        _ENTRY_MODULE,
        "replay-one",
        "--repository",
        str(repository),
        "--activity",
        str(ledger.root),
        "--ordinal",
        str(item.index),
        "--output-bytes",
        str(allowance.output_bytes),
        "--oracle-cpu-seconds",
        str(allowance.oracle_cpu_seconds),
    ]
    reason = None
    try:
        child = run_in_process_group_bounded(
            command,
            timeout=allowance.wall_seconds,
            max_capture_bytes=_ENTRY_CAPTURE_BYTES,
        )
        if child.returncode:
            reason = f"offline_worker_exit_{child.returncode}"
    except TimeoutExpired:
        reason = "offline_worker_wall_limit"
    receipt = output / "result.json"
    if not receipt.is_file():
        return AttemptOutcome(
            verdict=None,
            evidence=None,
            oracle_cpu_seconds=None,
            stop_reason=reason or "offline_worker_receipt_missing",
        )
    result = load_json_value(receipt)
    if (
        result["session_id"] != item.session.identity
        or result["gpu_initialized"]
    ):
        raise ValueError("E2 offline worker returned a mismatched receipt")
    source = (
        ledger.root / Phase.ONLINE / item.directory / "evidence/session.json"
    )
    incumbent = load_json_value(source)["original_evaluator_verdict"]
    return AttemptOutcome(
        verdict=incumbent,
        evidence=receipt,
        oracle_cpu_seconds=result["oracle_cpu_seconds"],
        stop_reason=result["stop_reason"] or reason,
    )


def _run_offline(
    repository: Path, freeze: VerifiedSuccessorFreeze
) -> dict[str, Any]:
    _require_device_free()
    ledger = SuccessorLedger(freeze.activity, freeze_sha256=freeze.sha256)
    if (ledger.root / Phase.OFFLINE).exists():
        raise FileExistsError(ledger.root / Phase.OFFLINE)
    closeout = ledger.root / _NATIVE_CLOSEOUT
    _verify_native_closeout(ledger, closeout)
    bind_gpu_handback(ledger, closeout)
    return execute_phase(
        ledger, Phase.OFFLINE, partial(_run_offline_attempt, repository, ledger)
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("verify", "online", "offline"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--repository", type=Path, required=True)
        sub.add_argument("--evidence-root", type=Path, required=True)
        sub.add_argument("--freeze", type=Path, required=True)
        if name == "online":
            sub.add_argument("--open-holdout", action="store_true")
    child = subparsers.add_parser("replay-one")
    child.add_argument("--repository", type=Path, required=True)
    child.add_argument("--activity", type=Path, required=True)
    child.add_argument("--ordinal", type=int, required=True)
    child.add_argument("--output-bytes", type=int, required=True)
    child.add_argument("--oracle-cpu-seconds", type=float, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Verify and dispatch in the selected immutable runtime."""
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "online" and not args.open_holdout:
        parser.error("online requires explicit --open-holdout authorization")
    if not sys.flags.safe_path:
        parser.error(
            "E2 execution requires Python -P to load the selected research code"
        )
    if os.environ.get(ENV_SOL_EXECBENCH_SANDBOXED) != "1":
        parser.error("E2 execution requires the hardened container")
    if args.command == "replay-one":
        return _replay_child(args)
    freeze = verify_frozen_successor(
        args.freeze.resolve(strict=True),
        repository=args.repository,
        capsule=Path(__file__).resolve().parents[3],
        evidence=args.evidence_root,
    )
    if args.command == "verify":
        print(freeze.sha256)
        return 0
    if args.command == "online":
        result = _run_online(
            args.repository, freeze, authorized=args.open_holdout
        )
    else:
        result = _run_offline(args.repository, freeze)
    print(result["state"])
    return int(result["state"] != "completed")


if __name__ == "__main__":
    sys.exit(main())

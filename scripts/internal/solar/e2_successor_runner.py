"""Connect compact evidence, bounded scratch and the original E2 evaluator."""

import math
import os
import secrets
import threading
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from scripts.internal.solar.e2_evidence_budget import retained_evidence
from scripts.internal.solar.e2_runner import SessionStaging, run_session
from scripts.internal.solar.e2_successor_campaign import (
    AttemptAllowance,
    AttemptOutcome,
)
from scripts.internal.solar.e2_successor_capture import (
    instrument_spooling_drivers,
)
from scripts.internal.solar.e2_successor_exposure import (
    verify_worker_commitment,
)
from scripts.internal.solar.e2_successor_ledger import PlannedSession
from sol_execbench.core.bench.judge.contracts import JudgeInstance
from sol_execbench.core.integrity import sha256_file

_SCRATCH = Path("/e2-scratch")
_SCRATCH_BYTES = 256 * 1024**2
_MIOPEN_USER_DB_PATH = "MIOPEN_USER_DB_PATH"
_SAMPLE_SECONDS = 0.025


def _scratch_usage() -> tuple[int, int]:
    info = os.statvfs(_SCRATCH)
    return (
        (info.f_blocks - info.f_bfree) * info.f_frsize,
        info.f_blocks * info.f_frsize,
    )


def require_empty_scratch() -> None:
    """Validate the dedicated frozen cache mount before any input exposure."""
    mounts = Path("/proc/self/mountinfo").read_text().splitlines()
    if not any(
        line.split()[4] == str(_SCRATCH)
        and line.split(" - ", maxsplit=1)[1].split()[0] == "tmpfs"
        for line in mounts
    ):
        raise ValueError("E2 scratch must be a dedicated tmpfs mount")
    if _scratch_usage() != (0, _SCRATCH_BYTES):
        raise ValueError("E2 scratch must be empty and have the frozen size")


def _sample_scratch(stop: threading.Event, metrics: dict[str, int]) -> None:
    while not stop.wait(_SAMPLE_SECONDS):
        metrics["sampled_peak_bytes"] = max(
            metrics["sampled_peak_bytes"], _scratch_usage()[0]
        )


def _restore_miopen_env(previous: str | None) -> None:
    if previous is None:
        os.environ.pop(_MIOPEN_USER_DB_PATH, None)
    else:
        os.environ[_MIOPEN_USER_DB_PATH] = previous


@contextmanager
def cold_session_workspace(output: Path) -> Iterator[dict[str, Any]]:
    """Route driver caches to a fresh kernel-bounded workspace for one arm."""
    require_empty_scratch()
    metrics = {
        "limit_bytes": _SCRATCH_BYTES,
        "start_bytes": 0,
        "sampled_peak_bytes": 0,
    }
    with ExitStack() as stack:
        stack.callback(require_empty_scratch)
        scratch = Path(stack.enter_context(TemporaryDirectory(dir=_SCRATCH)))
        for name in (".cache", ".tmp"):
            target = scratch / name.removeprefix(".")
            target.mkdir()
            link = output / name
            link.symlink_to(target, target_is_directory=True)
            stack.callback(link.unlink, missing_ok=True)
        database = scratch / "miopen-db"
        database.mkdir()
        previous = os.environ.get(_MIOPEN_USER_DB_PATH)
        stack.callback(_restore_miopen_env, previous)
        os.environ[_MIOPEN_USER_DB_PATH] = str(database)
        stop = threading.Event()
        thread = threading.Thread(
            target=_sample_scratch, args=(stop, metrics), daemon=True
        )
        thread.start()
        try:
            yield {"scratch": metrics}
        finally:
            stop.set()
            thread.join(timeout=1)
            if thread.is_alive():
                raise RuntimeError("E2 scratch sampling did not drain")
            metrics["sampled_peak_bytes"] = max(
                metrics["sampled_peak_bytes"], _scratch_usage()[0]
            )
    metrics["after_cleanup_bytes"] = 0


def _audit_capture(output: Path, result: dict[str, Any]) -> float | None:
    report = result.get("observations")
    if report is None:
        if not result["stop_reason"]:
            raise ValueError("E2 completed without a reference worker receipt")
        return None
    if report["publication_after_candidate_exit"] is not True:
        raise ValueError("E2 reference publication preceded candidate exit")
    if (output / "judge-observations/observations.json").is_symlink():
        raise ValueError("E2 reference receipt must be an owned file")
    total_bytes = 0
    for index, row in enumerate(report["observations"]):
        artifact = f"call-{index:03d}.npz"
        if row["artifact"] != artifact:
            raise ValueError("E2 actual-call archive order changed")
        archive = output / "judge-observations" / artifact
        if sha256_file(archive) != row["artifact_sha256"]:
            raise ValueError("E2 actual-call archive content changed")
        total_bytes += archive.stat().st_size
    if total_bytes != report["capture_bytes"]:
        raise ValueError("E2 actual-call archive byte total changed")
    cpu_seconds = report["oracle_cpu_seconds"]
    if cpu_seconds is None or not math.isfinite(cpu_seconds) or cpu_seconds < 0:
        raise ValueError(
            "E2 worker oracle CPU accounting is missing or invalid"
        )
    return float(cpu_seconds)


def run_development_attempt(
    repository: Path,
    item: PlannedSession,
    output: Path,
    allowance: AttemptAllowance,
    *,
    device: str = "cuda:0",
) -> AttemptOutcome:
    """Exercise the full online adapter without opening the held-out inputs."""
    return run_staged_attempt(
        repository,
        item,
        output,
        allowance,
        seed=item.session.case.require_development(),
        device=device,
    )


def run_staged_attempt(
    repository: Path,
    item: PlannedSession,
    output: Path,
    allowance: AttemptAllowance,
    *,
    seed: int,
    device: str = "cuda:0",
    holdout_commitment_sha256: str | None = None,
) -> AttemptOutcome:
    """Use the same bounded staging adapter on development and frozen inputs."""
    if item.session.case.instance is JudgeInstance.HOLDOUT:
        if device != "cuda:0":
            raise ValueError("E2 holdout requires the frozen GPU lane")
        verify_worker_commitment(output, holdout_commitment_sha256)
    result = run_session(
        repository,
        output,
        item.session,
        private_seed=seed,
        nonce=secrets.token_hex(32),
        remaining_seconds=allowance.wall_seconds,
        device=device,
        staging=SessionStaging(
            workspace=cold_session_workspace,
            instrument=partial(
                instrument_spooling_drivers,
                oracle_cpu_allowance_seconds=allowance.oracle_cpu_seconds,
                holdout_commitment_sha256=holdout_commitment_sha256,
            ),
        ),
    )
    spent = _audit_capture(output, result)
    used = retained_evidence(output)
    reason = result["stop_reason"]
    if max(used.logical_bytes, used.allocated_bytes) > allowance.output_bytes:
        reason = reason or "session_evidence_allowance_exceeded"
    return AttemptOutcome(
        verdict=result["original_evaluator_verdict"],
        evidence=output / "session.json",
        oracle_cpu_seconds=spent,
        stop_reason=reason,
    )

"""Opt-in worker-group RSS and evidence quotas for bounded research jobs."""

import os
import signal
import threading
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path

from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.process.process_observation import (
    session_member_process_groups,
)

_POLL_SECONDS = 0.1
_PROC = Path("/proc")


@dataclass(frozen=True, slots=True, kw_only=True)
class WorkerResourceBudget:
    """An explicit research cap, including mapper children in the worker group."""

    memory_bytes: int
    evidence_root: Path
    evidence_bytes: int

    def __post_init__(self) -> None:
        """Reject ineffective quotas before launching a worker."""
        if self.memory_bytes <= 0 or self.evidence_bytes <= 0:
            raise ValueError("worker resource budgets must be positive")


def process_group_rss(group: int) -> int:
    """Read current RSS of a process group without querying any GPU."""
    pages = 0
    for path in _PROC.iterdir():
        if not path.name.isdecimal():
            continue
        try:
            stat = (path / "stat").read_text()
            fields = stat[stat.rfind(")") + 2 :].split()
            if int(fields[2]) == group or int(fields[3]) == group:
                pages += int((path / "statm").read_text().split()[1])
        except FileNotFoundError, ProcessLookupError:
            continue
    return pages * os.sysconf("SC_PAGE_SIZE")


def evidence_size(root: Path, limit: int) -> int:
    """Stop scanning as soon as the byte quota is known to be exhausted."""
    size = 0
    for path in root.rglob("*"):
        try:
            if path.is_file() and not path.is_symlink():
                size += path.stat().st_size
        except FileNotFoundError:
            continue
        if size >= limit:
            break
    return size


@contextmanager
def enforce_worker_budget(
    budget: WorkerResourceBudget | None, failure_path: Path
) -> Iterator[None]:
    """Stop the entire isolated group on a quota breach; retain the exact cause."""
    if budget is None:
        yield
        return
    group = os.getpgrp()
    if group != os.getpid():
        raise ValueError(
            "resource guard requires an isolated process-group leader"
        )
    stop = threading.Event()
    thread = threading.Thread(
        target=_watch_budget,
        args=(budget, failure_path, group, stop),
        daemon=True,
    )
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=2)


def _watch_budget(
    budget: WorkerResourceBudget,
    failure: Path,
    group: int,
    stop: threading.Event,
) -> None:
    while not stop.is_set():
        try:
            if process_group_rss(group) >= budget.memory_bytes:
                reason = "memory_budget_exhausted"
            elif (
                evidence_size(budget.evidence_root, budget.evidence_bytes)
                >= budget.evidence_bytes
            ):
                reason = "evidence_budget_exhausted"
            else:
                stop.wait(_POLL_SECONDS)
                continue
        except OSError, ValueError, IndexError:
            reason = "resource_observation_failed"
        atomic_write_json_value(failure, {"reason": reason})
        extra_groups = session_member_process_groups(group) - {group}
        os.killpg(group, signal.SIGTERM)
        for pgid in sorted(extra_groups):
            with suppress(PermissionError, ProcessLookupError):
                os.killpg(pgid, signal.SIGTERM)
        return

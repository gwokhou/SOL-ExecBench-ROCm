"""Development-only E2 capture with a bounded, worker-private disk spool."""

import errno
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

from scripts.internal.solar.e2_contracts import PRIVATE_FILE
from scripts.internal.solar.e2_observation import Observation
from scripts.internal.solar.e2_service import ObservationService
from sol_execbench.core.bench.reference_service import serve_reference_worker
from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.process.termination import cleanup_on_termination

CAPTURE_BYTES = 1024**3
INTERRUPT_AFTER = 3
WORKER_DRAIN_SECONDS = 0.5
QUOTA_FAULT = "capture-quota"


class PilotObservationService(ObservationService):
    """Publish completed calls on close. No named raw files exist during requests."""

    def __init__(self, staging_dir: Path, **kwargs: Any) -> None:
        """Initialize private capture before the worker reports ready."""
        control = json.loads((staging_dir / PRIVATE_FILE).read_text())
        if control["instance"] not in {"d0", "d1"}:
            raise ValueError("pilot accepts only development instances")
        super().__init__(staging_dir, **kwargs)
        self.parent_pid = os.getppid()
        self.fault = control.get("pilot_fault")
        self.spool: list[tuple[int, dict[str, object]]] = []
        self.spool_bytes = 0
        self.spool_wall_seconds = 0.0

    def _record(self, observation: Observation) -> None:
        super()._record(observation)
        payload = self.buffers.pop()
        row = self.rows[-1]
        try:
            self._commit_capture(payload, row)
        except OSError as exc:
            self.stop_reason = (
                "pilot_capture_quota_exhausted"
                if exc.errno == errno.ENOSPC
                else "pilot_capture_write_failed"
            )
            # Abort instead of letting the protocol continue with an unrecorded call.
            os.killpg(os.getpgrp(), signal.SIGTERM)
            raise SystemExit(74) from exc

    def _commit_capture(self, payload: bytes, row: dict[str, object]) -> None:
        start = time.monotonic()
        size = self.spool_bytes + len(payload)
        at_fixture_limit = (
            self.fault == QUOTA_FAULT and len(self.spool) >= INTERRUPT_AFTER
        )
        if size > CAPTURE_BYTES or at_fixture_limit:
            raise OSError(
                errno.ENOSPC, "pilot anonymous capture quota exhausted"
            )
        fd = os.open(
            self.staging_dir, os.O_TMPFILE | os.O_RDWR | os.O_CLOEXEC, 0o600
        )
        owned = False
        try:
            with os.fdopen(os.dup(fd), "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fdatasync(handle.fileno())
            self.spool.append((fd, row))
            owned = True
            self.spool_bytes = size
        finally:
            if not owned:
                os.close(fd)
        atomic_write_json_value(
            self.staging_dir / "capture-progress.json",
            {
                "captured_prefix": len(self.spool),
                "anonymous_capture_bytes": size,
            },
        )
        self.spool_wall_seconds += time.monotonic() - start
        if len(self.spool) == INTERRUPT_AFTER and self.fault in {
            "term",
            "kill",
        }:
            signum = {"term": signal.SIGTERM, "kill": signal.SIGKILL}[
                self.fault
            ]
            os.killpg(os.getpgrp(), signum)

    def close(self) -> None:
        """Publish the completed prefix and release all private descriptors."""
        interrupted = isinstance(sys.exception(), SystemExit)
        try:
            if interrupted:
                self.stop_reason = (
                    self.stop_reason or "pilot_worker_interrupted"
                )
                self._wait_for_candidate_exit()
            self._publish()
        finally:
            for fd, _ in self.spool:
                os.close(fd)

    def _wait_for_candidate_exit(self) -> None:
        deadline = time.monotonic() + WORKER_DRAIN_SECONDS
        while os.getppid() == self.parent_pid:
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "pilot candidate exit was not confirmed before publication"
                )
            time.sleep(0.005)

    def _publish(self) -> None:
        start = time.monotonic()
        output = self.staging_dir / "judge-observations"
        output.mkdir(exist_ok=False)
        fd_directory = os.open("/proc/self/fd", os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            for index, (fd, row) in enumerate(self.spool):
                name = f"call-{index:03d}.npz"
                os.link(
                    str(fd),
                    output / name,
                    src_dir_fd=fd_directory,
                    follow_symlinks=True,
                )
                row["artifact"] = name
        finally:
            os.close(fd_directory)
        atomic_write_json_value(
            output / "observations.json",
            {
                "research_format": "e2-development-preflight-observations",
                "performance_authority": False,
                "observations": [row for _, row in self.spool],
                "stop_reason": self.stop_reason,
                "pending_correctness": self.pending_correctness is not None,
                "reference_calls": self.reference_calls,
                "reference_prepare_service_wall_seconds": self.service_seconds,
                "current_checks_seconds": self.current_checks_seconds,
                "capture_bytes": self.spool_bytes,
                "capture_wall_seconds": self.spool_wall_seconds,
                "publication_wall_seconds": time.monotonic() - start,
            },
        )


def serve_pilot_worker(staging_dir: str | Path, **kwargs: Any) -> None:
    """Use the incumbent authenticated protocol with research-only capture."""
    with cleanup_on_termination():
        serve_reference_worker(
            staging_dir, service_type=PilotObservationService, **kwargs
        )

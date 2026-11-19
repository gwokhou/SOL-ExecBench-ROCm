"""Publish bounded E2 research observations after the candidate has exited."""

import errno
import json
import os
import resource
import select
import signal
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np

from scripts.internal.solar.e2_archive_spool import EvidenceSpool
from scripts.internal.solar.e2_contracts import PRIVATE_FILE
from scripts.internal.solar.e2_observation import Observation
from scripts.internal.solar.e2_oracle_budget import (
    ORACLE_ALLOWANCE_FIELD,
    OracleCpuBudget,
)
from scripts.internal.solar.e2_service import ObservationService
from scripts.internal.solar.e2_staging import instrument_drivers
from scripts.internal.solar.e2_successor_exposure import (
    COMMITMENT_FIELD,
    protect_observation_process,
    verify_worker_commitment,
)
from sol_execbench.core.bench.reference_service import serve_reference_worker
from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.process.termination import cleanup_on_termination

CAPTURE_LIMIT_BYTES = 1024**3
_CANDIDATE_EXIT_TIMEOUT_MS = 2000
_STOCK_WORKER = (
    "from scripts.internal.solar.e2_service import serve_observed_worker "
    "as serve_reference_worker"
)
_SPOOL_WORKER = (
    "from scripts.internal.solar.e2_successor_capture import serve_spooling_worker "
    "as serve_reference_worker"
)
_WORKER_WAIT_HANDLE = (
    "    _candidate_environment[REFERENCE_PID_ENV] = str(_worker.pid)\n"
)


class SpoolingObservationService(ObservationService):
    """Keep actual-call archives in private files until the parent exits."""

    compact_output_evidence = True

    def __init__(self, staging_dir: Path, **kwargs: Any) -> None:
        """Check the input scope and capture the candidate lifecycle handle."""
        control = json.loads((staging_dir / PRIVATE_FILE).read_text())
        if control["instance"] == "h0":
            verify_worker_commitment(staging_dir, control.get(COMMITMENT_FIELD))
        elif control["instance"] not in {"d0", "d1"}:
            raise ValueError("unknown E2 capture instance")
        super().__init__(staging_dir, **kwargs)
        oracle_allowance = control.get(ORACLE_ALLOWANCE_FIELD)
        self._oracle_budget = (
            OracleCpuBudget(oracle_allowance)
            if oracle_allowance is not None
            else None
        )
        self._spool = EvidenceSpool(staging_dir, byte_limit=CAPTURE_LIMIT_BYTES)
        self._candidate = os.pidfd_open(os.getppid())
        self._capture_seconds = 0.0

    def _diagnose_observation(
        self, observation: Observation
    ) -> tuple[dict[str, object], dict[str, np.ndarray]]:
        with (
            self._oracle_budget.charge()
            if self._oracle_budget
            else nullcontext()
        ):
            return super()._diagnose_observation(observation)

    def _record(self, observation: Observation) -> None:
        super()._record(observation)
        payload = self.buffers.pop()
        started = time.monotonic()
        try:
            self._spool.capture(payload)
        except OSError as exc:
            self.rows.pop()
            self.stop_reason = (
                "capture_quota_exhausted"
                if exc.errno == errno.ENOSPC
                else "capture_write_failed"
            )
            os.killpg(os.getpgrp(), signal.SIGTERM)
            raise SystemExit(74) from exc
        atomic_write_json_value(
            self.staging_dir / "capture-progress.json",
            {
                "captured_prefix": self._spool.count,
                "anonymous_reserved_bytes": self._spool.reserved_bytes,
            },
        )
        self._capture_seconds += time.monotonic() - started

    def close(self) -> None:
        """Require candidate exit before exposing an archive or the input nonce."""
        try:
            poll = select.poll()
            poll.register(self._candidate, select.POLLIN)
            if not any(
                event & select.POLLIN
                for _, event in poll.poll(_CANDIDATE_EXIT_TIMEOUT_MS)
            ):
                raise RuntimeError(
                    "candidate exit unconfirmed at evidence publication"
                )
            self._publish()
        finally:
            self._spool.close()
            os.close(self._candidate)

    def _publish(self) -> None:
        output = self.staging_dir / "judge-observations"
        names = self._spool.publish(output)
        for row, name in zip(self.rows, names, strict=True):
            row["artifact"] = name
        atomic_write_json_value(
            output / "observations.json",
            {
                "research_format": "E2-private-observations-v1",
                "performance_authority": False,
                "private_nonce_after_channel_close": self.input_nonce,
                "publication_after_candidate_exit": True,
                "observations": self.rows,
                "stop_reason": self.stop_reason,
                "pending_correctness": self.pending_correctness is not None,
                "reference_calls": self.reference_calls,
                "reference_prepare_service_wall_seconds": self.service_seconds,
                "current_checks_seconds": self.current_checks_seconds,
                "reference_worker_peak_rss_bytes": resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss
                * 1024,
                "capture_wall_seconds": self._capture_seconds,
                "oracle_cpu_seconds": self._oracle_budget.spent_seconds
                if self._oracle_budget
                else None,
                "anonymous_reserved_bytes": self._spool.reserved_bytes,
                "capture_bytes": sum(
                    (output / name).stat().st_size for name in names
                ),
            },
        )


def serve_spooling_worker(staging_dir: str | Path, **kwargs: Any) -> None:
    """Activate capture in a separately staged research evaluation."""
    protect_observation_process()
    with cleanup_on_termination():
        serve_reference_worker(
            staging_dir, service_type=SpoolingObservationService, **kwargs
        )


def instrument_spooling_drivers(
    output: Path,
    *,
    oracle_cpu_allowance_seconds: float | None = None,
    holdout_commitment_sha256: str | None = None,
) -> None:
    """Keep the worker out of the candidate's shutdown wait; use a PID handle."""
    instrument_drivers(output)
    for name, original, updated in (
        ("reference_worker.py", _STOCK_WORKER, _SPOOL_WORKER),
        ("evaluation_orchestrator.py", _WORKER_WAIT_HANDLE, ""),
    ):
        path = output / name
        source = path.read_text()
        if source.count(original) != 1:
            raise ValueError(f"research publication seam changed: {name}")
        path.write_text(source.replace(original, updated))
    settings = {
        key: value
        for key, value in (
            (ORACLE_ALLOWANCE_FIELD, oracle_cpu_allowance_seconds),
            (COMMITMENT_FIELD, holdout_commitment_sha256),
        )
        if value is not None
    }
    if settings:
        path = output / PRIVATE_FILE
        control = json.loads(path.read_text())
        atomic_write_json_value(path, {**control, **settings})
        path.chmod(0o600)

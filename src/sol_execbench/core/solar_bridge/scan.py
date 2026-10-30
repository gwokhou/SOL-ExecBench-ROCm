# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Serial scan subprocess policy and bounded reports across the SOLAR boundary."""

from __future__ import annotations

import json
import subprocess
import sys
from enum import StrEnum
from pathlib import Path
from tempfile import TemporaryDirectory

from sol_execbench.core.process.subprocesses import run_in_process_group_bounded
from sol_execbench.core.solar_bridge.resource_policy import (
    available_formal_mapper_logical_cpu_count,
    formal_mapper_job_limit,
    formal_mapper_thread_count,
)
from solar.analysis.exploration_contracts import MAX_EXPLORATION_DOCUMENT_BYTES
from solar.analysis.scan_contracts import (
    SCAN_FILENAME,
    SCAN_WORKER_GRACE_SECONDS,
    ScanReference,
    ScanResult,
)
from solar.analysis.scan_reporting import report_scan
from solar.artifacts import sha256_file, stable_json_checksum
from solar.artifacts.bundle import read_regular_file
from solar.artifacts.json import write_json_artifact
from solar.errors import SolarError
from solar.pipeline.exploration import load_exploration_request
from solar.pipeline.scan_artifacts import load_scan
from solar.scan_contracts import ScanRequest
from solar.schema_versions import SOLAR_EXPLORATION_SCHEMA_VERSION

SCAN_WORKER_MODULE = "sol_execbench.core.solar_bridge.scan_worker"
MAX_SCAN_RESPONSE_BYTES = 65536
MAX_SCAN_PROCESS_OUTPUT_BYTES = 1024 * 1024


class ScanWorkerStatus(StrEnum):
    """Completion of the isolated attempt, independently of individual points."""

    COMPLETED = "completed"
    FAILED = "failed"


class ScanWorkerError(SolarError):
    """The isolated attempt has no accepted completion response."""

    reason_code = "scan_worker_failed"


def scan_execution_identity() -> dict[str, str]:
    """Bind scheduler implementation and prove one job fits its mapper policy."""
    threads = formal_mapper_thread_count()
    cpus = available_formal_mapper_logical_cpu_count() or 1
    jobs, _ = formal_mapper_job_limit(cpus, mapper_threads=threads)
    if threads > cpus or jobs < 1:
        raise ValueError("mapper thread policy exceeds the visible CPU budget")
    return {
        "runtime": "isolated_serial_worker",
        "adapter_sha256": sha256_file(Path(__file__)),
        "worker_sha256": sha256_file(
            Path(__file__).with_name("scan_worker.py")
        ),
        "logical_cpus": str(cpus),
        "mapper_threads": str(threads),
    }


def run_scan_worker(request: ScanRequest) -> ScanResult:
    """Apply an outer hard limit in addition to cooperative per-point deadlines."""
    execution = scan_execution_identity()
    with TemporaryDirectory(prefix="solar-scan-worker-") as directory:
        root = Path(directory)
        job, response = root / "request.json", root / "response.json"
        write_json_artifact(
            job,
            request.model_dump(mode="json"),
            max_bytes=MAX_EXPLORATION_DOCUMENT_BYTES,
        )
        command = [
            sys.executable,
            "-m",
            SCAN_WORKER_MODULE,
            str(job),
            str(response),
        ]
        try:
            completed = run_in_process_group_bounded(
                command,
                timeout=request.budget.total_seconds
                + SCAN_WORKER_GRACE_SECONDS,
                max_capture_bytes=MAX_SCAN_PROCESS_OUTPUT_BYTES,
            )
        except subprocess.TimeoutExpired as exc:
            raise ScanWorkerError(
                "scan worker exceeded its total time and cleanup grace; retain its attempt directory"
            ) from exc
        if not response.is_file():
            raise ScanWorkerError(
                f"scan worker has no response (exit {completed.returncode}); retain its attempt directory"
            )
        payload = json.loads(
            read_regular_file(response, max_bytes=MAX_SCAN_RESPONSE_BYTES)
        )
        return _accepted_scan(request, payload, completed.returncode, execution)


def _accepted_scan(
    request: ScanRequest,
    payload: object,
    returncode: int,
    execution: dict[str, str],
) -> ScanResult:
    if (
        not isinstance(payload, dict)
        or type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != SOLAR_EXPLORATION_SCHEMA_VERSION
    ):
        raise ScanWorkerError("scan response schema is not current")
    if payload.get("request_sha256") != stable_json_checksum(
        request.model_dump(mode="json")
    ):
        raise ScanWorkerError("scan response request identity differs")
    if returncode != 0 or payload.get("status") != ScanWorkerStatus.COMPLETED:
        raise ScanWorkerError(
            str(payload.get("message", "scan worker failed"))[:4096]
        )
    reference = ScanReference.model_validate(
        {
            "directory": request.job.projection.output_dir.resolve(),
            "record_sha256": payload.get("record_sha256"),
        }
    )
    result = load_scan(reference)
    if (
        result.identity_inputs.get("execution") != execution
        or result.identity_inputs.get("axes")
        != [axis.model_dump(mode="json") for axis in request.axes]
        or result.identity_inputs.get("view") != request.view
    ):
        raise ScanWorkerError("scan result runtime or grid identity differs")
    return result


def run_scan_report(path: Path) -> tuple[dict[str, object], Path]:
    """Keep missing points and explicit objective units in the public CLI report."""
    request = load_exploration_request(path, ScanRequest)
    result = run_scan_worker(request)
    return report_scan(
        request, result
    ), request.job.projection.output_dir / SCAN_FILENAME

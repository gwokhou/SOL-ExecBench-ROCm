# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Bounded subprocess adapter for fixed-instance SOLAR reuse."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from sol_execbench.core.process.subprocesses import run_in_process_group_bounded
from solar.analysis.demand_contracts import DEMAND_FILENAME
from solar.artifacts import stable_json_checksum
from solar.artifacts.bundle import read_regular_file
from solar.artifacts.document import ArtifactDocument, parse_yaml_artifact
from solar.artifacts.json import write_json_artifact
from solar.artifacts.names import ANALYSIS_FILENAME, REUSE_FILENAME
from solar.contracts import SolarAnalysisStatus
from solar.errors import SolarError
from solar.pipeline.reuse_artifacts import load_projection
from solar.reuse_contracts import ReuseJob, ReuseResult
from solar.schema_versions import SOLAR_REUSE_SCHEMA_VERSION

MAX_REUSE_REQUEST_BYTES = 2 * 1024 * 1024
MAX_REUSE_RESPONSE_BYTES = 65536
MAX_REUSE_PROCESS_OUTPUT_BYTES = 1024 * 1024
DEFAULT_REUSE_TIMEOUT_SECONDS = 900.0
REUSE_WORKER_MODULE = "sol_execbench.core.solar_bridge.reuse_worker"


@dataclass(frozen=True)
class ReuseReport:
    """Outer report keeps CLI dependencies behind the SOLAR bridge."""

    data: dict[str, object]
    artifacts: tuple[Path, ...]


def run_reuse_report(path: Path, *, timeout_seconds: float) -> ReuseReport:
    """Present complete views and eligibility from a bounded isolated analysis."""
    result = run_reuse_worker(
        load_reuse_job(path), timeout_seconds=timeout_seconds
    )
    metadata = ArtifactDocument(
        result.output_dir, result.analysis
    ).require_mapping("metadata")
    return ReuseReport(
        data={
            "scenario": result.record.scenario.value,
            "projection_sha256": result.record.projection_sha256,
            "sol_score_eligible": result.record.sol_score_eligible,
            "publication_eligible": result.record.publication_eligible,
            "views": metadata["views"],
            "restricted": result.restricted.model_dump(mode="json")
            if result.restricted is not None
            else None,
        },
        artifacts=tuple(
            result.output_dir / name
            for name in (
                ANALYSIS_FILENAME,
                REUSE_FILENAME,
                *((DEMAND_FILENAME,) if result.restricted is not None else ()),
            )
        ),
    )


class ReuseWorkerError(SolarError):
    """The isolated analysis attempt did not produce an accepted result."""

    reason_code = "reuse_worker_failed"


class ReuseWorkerTimeoutError(ReuseWorkerError):
    """The bounded worker exceeded its permitted wall time."""

    reason_code = "reuse_worker_timeout"


def load_reuse_job(path: Path) -> ReuseJob:
    """Require the current version before interpreting invocation fields."""
    document = parse_yaml_artifact(
        path,
        read_regular_file(path, max_bytes=MAX_REUSE_REQUEST_BYTES),
        max_bytes=MAX_REUSE_REQUEST_BYTES,
    )
    if (
        type(document.get("schema_version")) is not int
        or document.get("schema_version") != SOLAR_REUSE_SCHEMA_VERSION
    ):
        raise ValueError("reuse job schema is not current")
    return ReuseJob.model_validate(document.data)


def run_reuse_worker(
    job: ReuseJob, *, timeout_seconds: float = DEFAULT_REUSE_TIMEOUT_SECONDS
) -> ReuseResult:
    """Run one immutable job with bounded capture and process-group cleanup."""
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("reuse worker timeout must be finite and positive")
    with TemporaryDirectory(prefix="solar-reuse-worker-") as directory:
        root = Path(directory)
        request, response = root / "request.json", root / "response.json"
        write_json_artifact(
            request,
            job.model_dump(mode="json"),
            max_bytes=MAX_REUSE_REQUEST_BYTES,
        )
        command = [
            sys.executable,
            "-m",
            REUSE_WORKER_MODULE,
            str(request),
            str(response),
        ]
        try:
            completed = run_in_process_group_bounded(
                command,
                timeout=timeout_seconds,
                max_capture_bytes=MAX_REUSE_PROCESS_OUTPUT_BYTES,
            )
        except subprocess.TimeoutExpired as exc:
            raise ReuseWorkerTimeoutError(
                f"reuse worker exceeded {timeout_seconds:g} seconds"
            ) from exc
        if not response.is_file():
            raise ReuseWorkerError(
                f"reuse worker produced no response (exit {completed.returncode})"
            )
        payload = json.loads(
            read_regular_file(response, max_bytes=MAX_REUSE_RESPONSE_BYTES)
        )
        return _accepted_response(job, payload, completed.returncode)


def _accepted_response(
    job: ReuseJob, payload: object, returncode: int
) -> ReuseResult:
    if (
        not isinstance(payload, dict)
        or type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != SOLAR_REUSE_SCHEMA_VERSION
    ):
        raise ReuseWorkerError("reuse worker response schema is not current")
    if payload.get("job_sha256") != stable_json_checksum(
        job.model_dump(mode="json")
    ):
        raise ReuseWorkerError("reuse worker response job identity differs")
    if returncode != 0 or payload.get("status") != SolarAnalysisStatus.ANALYZED:
        raise ReuseWorkerError(
            str(payload.get("message", "reuse worker failed"))[:4096]
        )
    record_sha256, projection_sha256 = (
        payload.get("record_sha256"),
        payload.get("projection_sha256"),
    )
    if not isinstance(record_sha256, str) or not isinstance(
        projection_sha256, str
    ):
        raise ReuseWorkerError("reuse worker response lacks result identities")
    result = load_projection(
        job.projection.output_dir.resolve(),
        record_sha256=record_sha256,
        projection_sha256=projection_sha256,
        budget=job.source.budget,
    )
    if result.record.source_manifest_sha256 != job.source.manifest_sha256:
        raise ReuseWorkerError("reuse worker returned another source bundle")
    return result

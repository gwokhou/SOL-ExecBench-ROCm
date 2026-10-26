# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Bounded subprocess entry for original-oracle contraction search."""

from __future__ import annotations

import json
import subprocess
import sys
from enum import StrEnum
from pathlib import Path
from tempfile import TemporaryDirectory

from sol_execbench.core.process.subprocesses import run_in_process_group_bounded
from sol_execbench.core.solar_bridge.order_request import (
    OrderWorkerRequest,
    load_order_request,
)
from solar.analysis.order_contracts import (
    MAX_ORDER_REPORT_BYTES,
    ORDER_FILENAME,
    ORDER_WORKER_GRACE_SECONDS,
    OrderReference,
    OrderResult,
)
from solar.artifacts import sha256_file, stable_json_checksum
from solar.artifacts.bundle import read_regular_file
from solar.artifacts.json import write_json_artifact
from solar.errors import SolarError
from solar.pipeline.order_artifacts import load_order_result
from solar.schema_versions import SOLAR_ORDER_SCHEMA_VERSION

ORDER_WORKER_MODULE = "sol_execbench.core.solar_bridge.order_worker"
MAX_ORDER_RESPONSE_BYTES = 65536
MAX_ORDER_PROCESS_OUTPUT_BYTES = 1024 * 1024


class OrderWorkerStatus(StrEnum):
    """Process completion is separate from each candidate's semantic verdict."""

    COMPLETED = "completed"
    FAILED = "failed"


class OrderWorkerError(SolarError):
    """The child did not return an accepted, source-bound final record."""

    reason_code = "contraction_search_worker_failed"


def order_execution_identity() -> dict[str, str]:
    """Bind the adapter, exact-byte oracle loader and worker implementation."""
    root = Path(__file__).parent
    return {
        "runtime": "isolated_serial_cpu_worker",
        "adapter_sha256": sha256_file(Path(__file__)),
        "worker_sha256": sha256_file(root / "order_worker.py"),
        "oracle_loader_sha256": sha256_file(root / "order_request.py"),
    }


def run_order_worker(request: OrderWorkerRequest) -> OrderResult:
    """Bound runtime/output and clean the complete owned process group."""
    execution = order_execution_identity()
    with TemporaryDirectory(prefix="solar-order-worker-") as directory:
        root = Path(directory)
        job, response = root / "request.json", root / "response.json"
        write_json_artifact(
            job,
            request.model_dump(mode="json"),
            max_bytes=MAX_ORDER_REPORT_BYTES,
        )
        try:
            completed = run_in_process_group_bounded(
                [
                    sys.executable,
                    "-m",
                    ORDER_WORKER_MODULE,
                    str(job),
                    str(response),
                ],
                timeout=request.search.budget.total_seconds
                + ORDER_WORKER_GRACE_SECONDS,
                max_capture_bytes=MAX_ORDER_PROCESS_OUTPUT_BYTES,
            )
        except subprocess.TimeoutExpired as exc:
            raise OrderWorkerError(
                "search worker exceeded its total deadline and cleanup grace; retain the attempt directory"
            ) from exc
        if not response.is_file():
            raise OrderWorkerError(
                f"search worker has no completion response (exit {completed.returncode}); retain the attempt directory"
            )
        payload = json.loads(
            read_regular_file(response, max_bytes=MAX_ORDER_RESPONSE_BYTES)
        )
        return _accepted_order(
            request, payload, completed.returncode, execution
        )


def _accepted_order(
    request: OrderWorkerRequest,
    payload: object,
    code: int,
    execution: dict[str, str],
) -> OrderResult:
    if (
        not isinstance(payload, dict)
        or type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != SOLAR_ORDER_SCHEMA_VERSION
    ):
        raise OrderWorkerError("search response schema is not current")
    if payload.get("request_sha256") != stable_json_checksum(
        request.model_dump(mode="json")
    ):
        raise OrderWorkerError("search worker request identity differs")
    if code != 0 or payload.get("status") != OrderWorkerStatus.COMPLETED:
        raise OrderWorkerError(
            str(payload.get("message", "search worker failed"))[:4096]
        )
    reference = OrderReference.model_validate(
        {
            "directory": request.search.job.projection.output_dir.resolve(),
            "record_sha256": payload.get("record_sha256"),
        }
    )
    result = load_order_result(reference)
    if (
        result.identity_inputs.get("execution") != execution
        or result.identity_inputs.get("view") != request.search.view
        or result.identity_inputs.get("computation")
        != request.search.computation
    ):
        raise OrderWorkerError(
            "search result execution or model identity differs"
        )
    return result


def run_order_report(path: Path) -> tuple[dict[str, object], Path]:
    """Report the original baseline, full candidate denominator and all failures."""
    request = load_order_request(path)
    result = run_order_worker(request)
    return result.model_dump(
        mode="json"
    ), request.search.job.projection.output_dir / ORDER_FILENAME

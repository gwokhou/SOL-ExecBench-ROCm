# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Isolated entry point for a bounded, pinned SOLAR reuse job."""

from __future__ import annotations

import argparse
from pathlib import Path

from sol_execbench.core.solar_bridge.reuse import (
    MAX_REUSE_RESPONSE_BYTES,
    load_reuse_job,
)
from solar.analysis.interrupts import cleanup_on_termination
from solar.artifacts import sha256_file, stable_json_checksum
from solar.artifacts.json import write_json_artifact
from solar.artifacts.names import REUSE_FILENAME
from solar.contracts import SolarAnalysisStatus
from solar.reuse import ReuseSession
from solar.schema_versions import SOLAR_REUSE_SCHEMA_VERSION


def main() -> None:
    """Execute one validated invocation; unexpected invariant failures propagate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("response", type=Path)
    args = parser.parse_args()
    job = load_reuse_job(args.request)
    response: dict[str, object] = {
        "schema_version": SOLAR_REUSE_SCHEMA_VERSION,
        "job_sha256": stable_json_checksum(job.model_dump(mode="json")),
    }
    try:
        with cleanup_on_termination():
            result = ReuseSession(
                job.source, access_budget=job.access_budget
            ).project(job.projection)
        response.update(
            {
                "status": SolarAnalysisStatus.ANALYZED.value,
                "record_sha256": sha256_file(
                    result.output_dir / REUSE_FILENAME
                ),
                "projection_sha256": result.record.projection_sha256,
            }
        )
        exit_code = 0
    except (OSError, ValueError) as exc:
        response.update(
            {
                "status": SolarAnalysisStatus.FAILED.value,
                "message": str(exc)[:4096],
            }
        )
        exit_code = 1
    write_json_artifact(
        args.response, response, max_bytes=MAX_REUSE_RESPONSE_BYTES
    )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()

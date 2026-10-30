# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Isolated entry point for one bounded SOLAR architecture scan."""

from __future__ import annotations

import argparse
from pathlib import Path

from sol_execbench.core.solar_bridge.scan import (
    MAX_SCAN_RESPONSE_BYTES,
    ScanWorkerStatus,
    scan_execution_identity,
)
from solar.analysis.scan_contracts import SCAN_FILENAME
from solar.artifacts import sha256_file, stable_json_checksum
from solar.artifacts.json import write_json_artifact
from solar.explore import scan_profiles
from solar.pipeline.exploration import load_exploration_request
from solar.scan_contracts import ScanRequest
from solar.schema_versions import SOLAR_EXPLORATION_SCHEMA_VERSION


def main() -> None:
    """Preserve expected input failures; unexpected invariants exit unsuccessfully."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("response", type=Path)
    args = parser.parse_args()
    request = load_exploration_request(args.request, ScanRequest)
    payload: dict[str, object] = {
        "schema_version": SOLAR_EXPLORATION_SCHEMA_VERSION,
        "request_sha256": stable_json_checksum(request.model_dump(mode="json")),
    }
    try:
        scan_profiles(request, execution_identity=scan_execution_identity())
        payload.update(
            {
                "status": ScanWorkerStatus.COMPLETED,
                "record_sha256": sha256_file(
                    request.job.projection.output_dir / SCAN_FILENAME
                ),
            }
        )
        code = 0
    except (OSError, ValueError) as exc:
        payload.update(
            {"status": ScanWorkerStatus.FAILED, "message": str(exc)[:4096]}
        )
        code = 1
    write_json_artifact(
        args.response, payload, max_bytes=MAX_SCAN_RESPONSE_BYTES
    )
    raise SystemExit(code)


if __name__ == "__main__":
    main()

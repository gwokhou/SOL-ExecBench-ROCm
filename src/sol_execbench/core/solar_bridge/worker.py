# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Subprocess entry point for isolated SOLAR analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

from sol_execbench.core.data.json_utils import load_json_value
from sol_execbench.core.solar_bridge.analyzer import analyze_workload
from sol_execbench.core.solar_bridge.models import (
    SolarAnalysisOutcome,
    SolarWorkerRequest,
)
from sol_execbench.core.solar_bridge.worker_io import write_worker_response


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("response", type=Path)
    args = parser.parse_args()
    request_data = load_json_value(args.request)
    request = SolarWorkerRequest.from_dict(request_data)
    try:
        outcome = analyze_workload(
            problem_dir=request.problem_dir,
            workload_uuid=request.workload_uuid,
            output_dir=request.output_dir,
            device=request.device,
            orojenesis_home=request.orojenesis_home,
        )
    except Exception as exc:
        outcome = SolarAnalysisOutcome(
            status="failed",
            analysis_id=request.workload_uuid,
            stage="outer_bridge",
            reason_code="bridge_failed",
            message=str(exc),
        )
    fallback = SolarAnalysisOutcome(
        status="failed",
        analysis_id=request.workload_uuid,
        stage="outer_bridge",
        reason_code="worker_response_failed",
        message="worker response serialization failed",
    )
    response_written = write_worker_response(
        args.response, outcome.to_dict(), fallback.to_dict()
    )
    raise SystemExit(0 if outcome.status == "analyzed" and response_written else 1)


if __name__ == "__main__":
    main()

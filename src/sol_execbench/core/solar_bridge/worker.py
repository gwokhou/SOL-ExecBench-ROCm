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
    SolarAnalysisStatus,
    SolarStage,
    SolarWorkerRequest,
)
from sol_execbench.core.solar_bridge.worker_io import write_worker_response

_SOLAR_WORKER_OOM_SCORE_ADJUSTMENT = 500
_OOM_SCORE_ADJUSTMENT_PATH = Path("/proc/self/oom_score_adj")


def _prefer_worker_as_oom_victim(
    path: Path = _OOM_SCORE_ADJUSTMENT_PATH,
) -> None:
    """Protect the release parent by making this worker the OOM victim."""
    try:
        current = int(path.read_text(encoding="utf-8").strip())
        target = max(current, _SOLAR_WORKER_OOM_SCORE_ADJUSTMENT)
        path.write_text(f"{target}\n", encoding="utf-8")
        observed = int(path.read_text(encoding="utf-8").strip())
    except (OSError, UnicodeError, ValueError) as exc:
        raise RuntimeError(
            "SOLAR parallel worker could not configure OOM isolation",
        ) from exc
    if observed < target:
        raise RuntimeError(
            "SOLAR parallel worker OOM isolation was not applied",
        )


def main() -> None:
    """Run one isolated SOLAR analysis worker request."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("response", type=Path)
    args = parser.parse_args()
    request_data = load_json_value(args.request)
    request = SolarWorkerRequest.from_dict(request_data)
    try:
        if request.device_stage_lock_path is not None:
            _prefer_worker_as_oom_victim()
        outcome = analyze_workload(
            problem_dir=request.problem_dir,
            workload_uuid=request.workload_uuid,
            output_dir=request.output_dir,
            device=request.device,
            orojenesis_home=request.orojenesis_home,
            ir_path=request.ir_path,
            device_stage_lock_path=request.device_stage_lock_path,
            device_stage_lock_timeout_seconds=(
                request.device_stage_lock_timeout_seconds
            ),
        )
    except Exception as exc:  # noqa: BLE001 -- isolated worker boundary
        outcome = SolarAnalysisOutcome(
            status=SolarAnalysisStatus.FAILED,
            analysis_id=request.workload_uuid,
            ir_path=request.ir_path,
            stage=SolarStage.OUTER_BRIDGE,
            reason_code="bridge_failed",
            message=str(exc),
        )
    fallback = SolarAnalysisOutcome(
        status=SolarAnalysisStatus.FAILED,
        analysis_id=request.workload_uuid,
        ir_path=request.ir_path,
        stage=SolarStage.OUTER_BRIDGE,
        reason_code="worker_response_failed",
        message="worker response serialization failed",
    )
    response_written = write_worker_response(
        args.response,
        outcome.to_dict(),
        fallback.to_dict(),
    )
    raise SystemExit(
        0
        if outcome.status is SolarAnalysisStatus.ANALYZED and response_written
        else 1,
    )


if __name__ == "__main__":
    main()

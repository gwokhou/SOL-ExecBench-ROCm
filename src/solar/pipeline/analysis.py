# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Atomic execution behind the public SOLAR API boundary."""

from __future__ import annotations

import fcntl
import time
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from pathlib import Path

from solar.contracts import (
    AnalysisExecutionPolicy,
    AnalysisRequest,
    SolarStage,
)
from solar.ir.contracts import IRGraphArtifact
from solar.pipeline.stages import (
    analyze_request_graph,
    convert_request_graph,
    extract_request_graph,
    verify_request_graph,
    workflow_reason_code,
)
from solar.rocm.architecture import ArchitectureProfile
from solar.types import DynamicValue


@dataclass(frozen=True)
class PipelineResult:
    """Graph, IR, verification, and analysis output."""

    ir_graph: IRGraphArtifact
    analysis: dict[str, DynamicValue]


class PipelineStageError(RuntimeError):
    """Preserve the precise public stage for a pipeline failure."""

    def __init__(self, stage: SolarStage, error: Exception) -> None:
        """Wrap an error with the stage that produced it."""
        super().__init__(str(error))
        self.stage = stage
        self.error = error


@contextmanager
def _exclusive_device_stage(
    policy: AnalysisExecutionPolicy,
) -> Iterator[None]:
    """Serialize device-backed stages without affecting published evidence."""
    path = policy.device_stage_lock_path
    if path is None:
        yield
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + policy.device_stage_lock_timeout_seconds
    with path.open("a+", encoding="utf-8") as handle:
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "SOLAR device-stage lock wait exceeded "
                        f"{policy.device_stage_lock_timeout_seconds:g} seconds",
                    ) from None
                time.sleep(0.05)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _device_stage_guard(
    request: AnalysisRequest,
) -> AbstractContextManager[None]:
    """Return the configured guard for extraction through verification."""
    return _exclusive_device_stage(request.execution_policy)


def run_pipeline(
    request: AnalysisRequest,
    profile: ArchitectureProfile,
    staging: Path,
) -> PipelineResult:
    """Run the graph, IR, verification, and analysis stages."""
    stage = SolarStage.GRAPH_EXTRACTION
    try:
        with _device_stage_guard(request):
            operator = extract_request_graph(request, staging)
            stage = SolarStage.IR_CONVERSION
            ir_graph = convert_request_graph(request, operator, staging)
            stage = SolarStage.CONVERSION_VERIFICATION
            verify_request_graph(
                request,
                ir_graph,
                staging / "conversion-attestation.yaml",
            )
            cleanup = request.execution_policy.device_stage_cleanup
            if cleanup is not None:
                cleanup()
        stage = SolarStage.FORMAL_ANALYSIS
        analysis = analyze_request_graph(request, profile, staging, ir_graph)
        return PipelineResult(ir_graph, analysis)
    except Exception as exc:
        raise PipelineStageError(stage, exc) from exc


def pipeline_reason_code(stage: SolarStage, exc: Exception) -> str:
    """Map implementation errors onto stable public reason codes."""
    return workflow_reason_code(stage, exc)


__all__ = [
    "PipelineStageError",
    "pipeline_reason_code",
    "run_pipeline",
]

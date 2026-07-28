# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Route-independent execution behind the public SOLAR API boundary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from solar.common.types import DynamicValue
from solar.contracts import AnalysisRequest, SolarStage
from solar.ir.contracts import IRGraphArtifact
from solar.rocm.architecture import ArchitectureProfile
from solar.schema_versions import SOLAR_ANALYSIS_SCHEMA_VERSION
from solar.workflow import (
    analyze_request_graph,
    convert_request_graph,
    extract_request_graph,
    verify_request_graph,
    workflow_reason_code,
)


@dataclass(frozen=True)
class PipelineResult:
    """Route-specific graph, IR, verification, and analysis output."""

    ir_graph: IRGraphArtifact
    analysis: dict[str, DynamicValue]


class PipelineStageError(RuntimeError):
    """Preserve the precise public stage for a route helper failure."""

    def __init__(self, stage: SolarStage, error: Exception) -> None:
        """Wrap a route error with the stage that produced it."""
        super().__init__(str(error))
        self.stage = stage
        self.error = error


def run_route_pipeline(
    request: AnalysisRequest,
    profile: ArchitectureProfile,
    staging: Path,
) -> PipelineResult:
    """Run the graph, IR, verification, and analysis stages for a route."""
    stage = SolarStage.GRAPH_EXTRACTION
    try:
        operator = extract_request_graph(request, staging)
        stage = SolarStage.IR_CONVERSION
        ir_graph = convert_request_graph(request, operator, staging)
        stage = SolarStage.CONVERSION_VERIFICATION
        verify_request_graph(
            request,
            ir_graph,
            staging / "conversion-attestation.yaml",
        )
        stage = SolarStage.FORMAL_ANALYSIS
        analysis = analyze_request_graph(request, profile, staging, ir_graph)
        return PipelineResult(ir_graph, analysis)
    except Exception as exc:
        raise PipelineStageError(stage, exc) from exc


def pipeline_reason_code(stage: SolarStage, exc: Exception) -> str:
    """Map route implementation errors onto stable public reason codes."""
    return workflow_reason_code(stage, exc)


__all__ = [
    "SOLAR_ANALYSIS_SCHEMA_VERSION",
    "ArchitectureProfile",
    "PipelineStageError",
    "pipeline_reason_code",
    "run_route_pipeline",
]

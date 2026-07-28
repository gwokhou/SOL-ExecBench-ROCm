# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Shared graph, IR, verification, and analysis workflow stages."""

from __future__ import annotations

from pathlib import Path

from solar.contracts import SolarStage
from solar.errors import SolarError
from solar.graph.contracts import OperatorGraphArtifact
from solar.graph.extraction import extract_operator_graph
from solar.ir.contracts import (
    IRAnalysisRequest,
    IRConversionRequest,
    IRGraphArtifact,
)
from solar.ir.conversion import convert_operator_graph
from solar.ir.registry import ir_lifecycle
from solar.rocm.architecture import ArchitectureProfile
from solar.types import DynamicValue


def extract_request_graph(
    request: IRConversionRequest,
    staging: Path,
) -> OperatorGraphArtifact:
    """Extract an operator graph using the requested backend."""
    return extract_operator_graph(
        request.reference,
        tuple(request.input_factory(request.trace_seed)),
        device=request.device,
        output_dir=staging,
        name=request.analysis_id,
        extraction_kind=request.extraction_kind,
    )


def convert_request_graph(
    request: IRConversionRequest,
    operator: OperatorGraphArtifact,
    staging: Path,
) -> IRGraphArtifact:
    """Convert an operator graph through the selected IR backend."""
    return convert_operator_graph(
        operator,
        output_dir=staging,
        ir_kind=request.ir_kind,
    )


def verify_request_graph(
    request: IRConversionRequest,
    graph: IRGraphArtifact,
    output_path: Path,
) -> None:
    """Verify callable equivalence through the selected IR implementation."""
    _ = graph.document
    ir_lifecycle(graph.kind).verify(request, graph, output_path)


def analyze_request_graph(
    request: IRAnalysisRequest,
    profile: ArchitectureProfile,
    staging: Path,
    graph: IRGraphArtifact,
) -> dict[str, DynamicValue]:
    """Run formal analysis through the selected IR implementation."""
    _ = graph.document
    return ir_lifecycle(graph.kind).analyze(
        request,
        profile,
        staging,
        graph,
    )


def workflow_reason_code(stage: SolarStage, exc: Exception) -> str:
    """Map concrete workflow exceptions onto stable public reason codes."""
    if isinstance(exc, SolarError):
        return exc.reason_code
    return f"{stage}_failed"


__all__ = [
    "analyze_request_graph",
    "convert_request_graph",
    "extract_request_graph",
    "verify_request_graph",
    "workflow_reason_code",
]

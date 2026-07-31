# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Shared graph, IR, verification, and analysis workflow stages."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from solar.contracts import SolarStage
from solar.errors import SolarError
from solar.graph.contracts import OperatorGraphArtifact
from solar.graph.extraction import extract_operator_graph
from solar.ir.contracts import (
    IRConversionRequest,
    IRGraphArtifact,
)
from solar.ir.conversion import convert_operator_graph
from solar.ir.registry import ir_backend
from solar.rocm.architecture import ArchitectureProfile
from solar.types import DynamicValue
from solar.verification import verify_callable_conversion
from solar.verification.contracts import VerificationPolicy
from solar.verification.registry import verification_backend


class VerificationWorkflowRequest(IRConversionRequest, Protocol):
    """Request fields consumed by callable-to-IR verification."""

    @property
    def reference_name(self) -> str: ...

    @property
    def reference_sha256(self) -> str: ...

    @property
    def verification(self) -> VerificationPolicy: ...


class AnalysisWorkflowRequest(VerificationWorkflowRequest, Protocol):
    """Request fields consumed by formal analysis."""

    @property
    def precision(self) -> str: ...

    @property
    def require_orojenesis(self) -> bool: ...

    @property
    def orojenesis_home(self) -> str | Path | None: ...


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
    request: VerificationWorkflowRequest,
    graph: IRGraphArtifact,
    output_path: Path,
) -> None:
    """Verify callable equivalence through the selected IR implementation."""
    verify_callable_conversion(
        reference=request.reference,
        input_factory=request.input_factory,
        reference_name=request.reference_name,
        reference_sha256=request.reference_sha256,
        graph_path=graph.path,
        graph=graph.document,
        output_path=output_path,
        policy=request.verification,
        backend=verification_backend(graph.kind),
    )


def analyze_request_graph(
    request: AnalysisWorkflowRequest,
    profile: ArchitectureProfile,
    staging: Path,
    graph: IRGraphArtifact,
) -> dict[str, DynamicValue]:
    """Run formal analysis through the selected IR implementation."""
    from solar.analysis.graph_analyzer import IRGraphAnalyzer, OrojenesisRunner

    _ = graph.document
    backend = ir_backend(graph.kind)
    runner = (
        OrojenesisRunner(request.orojenesis_home)
        if request.require_orojenesis or request.orojenesis_home is not None
        else None
    )
    result = IRGraphAnalyzer(validator=backend.validate).analyze_graph(
        graph.path,
        staging,
        precision=request.precision,
        copy_graph=False,
        strict=True,
        architecture=profile,
        orojenesis_runner=runner,
        require_orojenesis=request.require_orojenesis,
    )
    if result is None:
        raise RuntimeError("strict graph analysis produced no artifact")
    return result


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

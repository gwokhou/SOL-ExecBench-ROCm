# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Shared graph, IR, verification, and analysis workflow stages."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

from solar.common.types import DynamicValue
from solar.contracts import SolarStage
from solar.graph.contracts import OperatorGraphArtifact
from solar.graph.extraction import extract_operator_graph
from solar.ir.contracts import IRGraphArtifact, IRKind
from solar.ir.conversion import convert_operator_graph
from solar.rocm.architecture import ArchitectureProfile
from solar.routes import Route, route_spec


class ConversionWorkflowRequest(Protocol):
    """Read-only request boundary shared by conversion workflows."""

    @property
    def analysis_id(self) -> str: ...

    @property
    def reference(self) -> Callable[..., DynamicValue]: ...

    @property
    def input_factory(self) -> Callable[[int], Sequence[DynamicValue]]: ...

    @property
    def reference_name(self) -> str: ...

    @property
    def reference_sha256(self) -> str: ...

    @property
    def route(self) -> Route | str: ...

    @property
    def representation(self) -> IRKind | str: ...

    @property
    def device(self) -> str: ...

    @property
    def trace_seed(self) -> int: ...

    @property
    def verification_seeds(self) -> tuple[int, ...]: ...

    @property
    def atol(self) -> float: ...

    @property
    def rtol(self) -> float: ...

    @property
    def required_matched_ratio(self) -> float: ...

    @property
    def max_error_cap(self) -> float | None: ...

    @property
    def allow_negative_inf(self) -> bool: ...


class WorkflowRequest(ConversionWorkflowRequest, Protocol):
    """Read-only request boundary for formal analysis."""

    @property
    def precision(self) -> str: ...

    @property
    def require_orojenesis(self) -> bool: ...

    @property
    def orojenesis_home(self) -> str | Path | None: ...


def extract_request_graph(
    request: ConversionWorkflowRequest,
    staging: Path,
) -> OperatorGraphArtifact:
    """Extract an operator graph using the request's declarative route."""
    return extract_operator_graph(
        request.reference,
        tuple(request.input_factory(request.trace_seed)),
        device=request.device,
        output_dir=staging,
        name=request.analysis_id,
        extraction=route_spec(request.route).extraction,
    )


def convert_request_graph(
    request: ConversionWorkflowRequest,
    operator: OperatorGraphArtifact,
    staging: Path,
) -> IRGraphArtifact:
    """Convert an operator graph through the selected IR backend."""
    return convert_operator_graph(
        operator,
        output_dir=staging,
        representation=request.representation,
    )


def verify_request_graph(
    request: ConversionWorkflowRequest,
    graph: IRGraphArtifact,
    output_path: Path,
) -> None:
    """Verify callable equivalence through the selected IR implementation."""
    _VERIFIERS[graph.kind](request, graph.path, output_path)


def analyze_request_graph(
    request: WorkflowRequest,
    profile: ArchitectureProfile,
    staging: Path,
    graph: IRGraphArtifact,
) -> dict[str, DynamicValue]:
    """Run formal analysis through the selected IR implementation."""
    return _ANALYZERS[graph.kind](request, profile, staging, graph.path)


def workflow_reason_code(stage: SolarStage, exc: Exception) -> str:
    """Map concrete workflow exceptions onto stable public reason codes."""
    from solar.analysis.graph_analyzer import OrojenesisError
    from solar.analysis.nvlabs.orojenesis import (
        OrojenesisError as NVLabsOrojenesisError,
    )
    from solar.verification import VerificationError
    from solar.verification.nvlabs_einsum import (
        VerificationError as NVLabsVerificationError,
    )

    if isinstance(exc, (OrojenesisError, NVLabsOrojenesisError)):
        return "toolchain_unavailable"
    if isinstance(exc, (VerificationError, NVLabsVerificationError)):
        return "conversion_not_proven"
    return f"{stage}_failed"


def _verify_aten(
    request: ConversionWorkflowRequest,
    graph_path: Path,
    output_path: Path,
) -> None:
    from solar.verification import (
        VerificationPolicy,
        verify_callable_conversion,
    )

    verify_callable_conversion(
        reference=request.reference,
        input_factory=request.input_factory,
        reference_name=request.reference_name,
        reference_sha256=request.reference_sha256,
        graph_path=graph_path,
        output_path=output_path,
        policy=VerificationPolicy(
            atol=request.atol,
            rtol=request.rtol,
            required_matched_ratio=request.required_matched_ratio,
            max_error_cap=request.max_error_cap,
            allow_negative_inf=request.allow_negative_inf,
            seeds=request.verification_seeds,
            device=request.device,
        ),
    )


def _verify_nvlabs(
    request: ConversionWorkflowRequest,
    graph_path: Path,
    output_path: Path,
) -> None:
    from solar.verification.nvlabs_einsum import verify_callable_conversion

    verify_callable_conversion(
        reference=request.reference,
        input_factory=request.input_factory,
        reference_name=request.reference_name,
        reference_sha256=request.reference_sha256,
        graph_path=graph_path,
        output_path=output_path,
        atol=request.atol,
        rtol=request.rtol,
        required_matched_ratio=request.required_matched_ratio,
        max_error_cap=request.max_error_cap,
        allow_negative_inf=request.allow_negative_inf,
        seeds=request.verification_seeds,
        device=request.device,
    )


def _analyze_aten(
    request: WorkflowRequest,
    profile: ArchitectureProfile,
    staging: Path,
    graph_path: Path,
) -> dict[str, DynamicValue]:
    from solar.analysis.graph_analyzer import IRGraphAnalyzer, OrojenesisRunner

    runner = (
        OrojenesisRunner(request.orojenesis_home)
        if request.require_orojenesis or request.orojenesis_home is not None
        else None
    )
    result = IRGraphAnalyzer().analyze_graph(
        graph_path,
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


def _analyze_nvlabs(
    request: WorkflowRequest,
    profile: ArchitectureProfile,
    staging: Path,
    graph_path: Path,
) -> dict[str, DynamicValue]:
    from solar.analysis.nvlabs.graph_analyzer import EinsumGraphAnalyzer
    from solar.analysis.nvlabs.orojenesis import (
        OrojenesisRunner as NVLabsOrojenesisRunner,
    )

    runner = (
        NVLabsOrojenesisRunner(request.orojenesis_home)
        if request.require_orojenesis or request.orojenesis_home is not None
        else None
    )
    result = EinsumGraphAnalyzer().analyze_graph(
        graph_path,
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


_VERIFIERS = {
    IRKind.ATEN: _verify_aten,
    IRKind.NVLABS_EINSUM: _verify_nvlabs,
}
_ANALYZERS = {
    IRKind.ATEN: _analyze_aten,
    IRKind.NVLABS_EINSUM: _analyze_nvlabs,
}


__all__ = [
    "analyze_request_graph",
    "convert_request_graph",
    "extract_request_graph",
    "verify_request_graph",
    "workflow_reason_code",
]

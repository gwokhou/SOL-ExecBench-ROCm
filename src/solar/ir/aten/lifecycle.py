# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Complete lifecycle registration for the ATen IR."""

from __future__ import annotations

from pathlib import Path

from solar.graph.contracts import ExtractionKind
from solar.ir.aten.conversion import convert_operator_graph, validate_aten_graph
from solar.ir.contracts import (
    IRAnalysisRequest,
    IRConversionRequest,
    IRGraphArtifact,
    IRKind,
    IRLifecycle,
)
from solar.rocm.architecture import ArchitectureProfile
from solar.types import DynamicValue
from solar.verification.aten import execute_aten_layer


def _verify(
    request: IRConversionRequest,
    graph: IRGraphArtifact,
    output_path: Path,
) -> None:
    from solar.verification import verify_callable_conversion

    verify_callable_conversion(
        reference=request.reference,
        input_factory=request.input_factory,
        reference_name=request.reference_name,
        reference_sha256=request.reference_sha256,
        graph_path=graph.path,
        graph=graph.document,
        output_path=output_path,
        policy=request.verification,
        lifecycle=lifecycle,
    )


def _analyze(
    request: IRAnalysisRequest,
    profile: ArchitectureProfile,
    staging: Path,
    graph: IRGraphArtifact,
) -> dict[str, DynamicValue]:
    from solar.analysis.graph_analyzer import IRGraphAnalyzer, OrojenesisRunner

    runner = (
        OrojenesisRunner(request.orojenesis_home)
        if request.require_orojenesis or request.orojenesis_home is not None
        else None
    )
    result = IRGraphAnalyzer(validator=lifecycle.validate).analyze_graph(
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


lifecycle = IRLifecycle(
    kind=IRKind.ATEN,
    extractions=frozenset({ExtractionKind.MAKE_FX_REFERENCE}),
    validate=validate_aten_graph,
    convert=convert_operator_graph,
    execute=execute_aten_layer,
    verify=_verify,
    analyze=_analyze,
)


__all__ = ["lifecycle"]

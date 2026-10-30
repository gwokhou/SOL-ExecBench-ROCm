# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Invoke SOLAR and enforce the requested bound acceptance policy."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from sol_execbench.core.solar_bridge.formal_device import (
    FORMAL_ARCHITECTURE,
    device_execution_policy,
)
from sol_execbench.core.solar_bridge.models import (
    SolarAnalysisOutcome,
    SolarAnalysisStatus,
    SolarStage,
    formal_precision_for_definition,
)
from sol_execbench.core.solar_bridge.semantic_metadata import (
    performance_analysis_metadata,
)
from sol_execbench.core.solar_bridge.workload_context import (
    SolarWorkloadContext,
    solar_conversion_request,
)
from solar.ir.contracts import DEFAULT_IR_PATH, IRPath

if TYPE_CHECKING:
    from solar.api import AnalysisFailure, AnalysisResult


def invoke_solar(
    *,
    context: SolarWorkloadContext,
    output_dir: Path,
    device: str,
    orojenesis_home: str | Path | None,
    ir_path: IRPath = DEFAULT_IR_PATH,
    architecture: str = FORMAL_ARCHITECTURE,
    formal: bool = True,
    device_stage_lock_path: str | Path | None = None,
    device_stage_lock_timeout_seconds: float = 14_400.0,
) -> SolarAnalysisOutcome:
    """Build a SOLAR request and adapt its accepted result to ExecBench."""
    from solar.api import AnalysisRequest, analyze

    definition = context.definition
    request = AnalysisRequest(
        conversion=solar_conversion_request(context, device, ir_path),
        architecture=architecture,
        output_dir=output_dir,
        precision=formal_precision_for_definition(definition),
        require_orojenesis=formal,
        require_verified_audit=formal,
        orojenesis_home=orojenesis_home,
        analysis_metadata=performance_analysis_metadata(context),
        execution_policy=device_execution_policy(
            device,
            device_stage_lock_path,
            device_stage_lock_timeout_seconds,
        ),
    )
    return _analysis_outcome(
        analyze(request), ir_path=request.ir_path, formal=formal
    )


def _analysis_outcome(
    result: AnalysisFailure | AnalysisResult,
    *,
    ir_path: IRPath,
    formal: bool,
) -> SolarAnalysisOutcome:
    """Map SOLAR results and discard bounds rejected by formal policy."""
    from solar.api import AnalysisFailure

    if isinstance(result, AnalysisFailure):
        return SolarAnalysisOutcome(
            status=result.status,
            analysis_id=result.analysis_id,
            ir_path=result.ir_path,
            stage=result.stage,
            reason_code=result.reason_code,
            message=result.message,
        )
    outcome = SolarAnalysisOutcome(
        status=result.status,
        analysis_id=result.analysis_id,
        ir_path=result.ir_path,
        output_dir=str(result.output_dir),
        architecture_sha256=result.architecture_sha256,
        lower_bound_seconds=result.bound.seconds,
        bound_kind=result.bound.kind,
        limiting_resource=result.bound.limiting_resource,
        artifacts=tuple(artifact.__dict__ for artifact in result.artifacts),
        publication_eligible=result.publication_eligible,
    )
    if formal and not outcome.is_formal_publication:
        shutil.rmtree(result.output_dir, ignore_errors=True)
        return SolarAnalysisOutcome(
            status=SolarAnalysisStatus.FAILED,
            analysis_id=result.analysis_id,
            ir_path=ir_path,
            stage=SolarStage.FORMAL_ACCEPTANCE,
            reason_code="non_formal_bound",
            message="SOLAR formal bridge rejected a non-publication result",
        )
    return outcome

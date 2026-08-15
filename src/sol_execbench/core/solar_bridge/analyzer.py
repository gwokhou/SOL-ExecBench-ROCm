# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""The sole in-process adapter from SOL ExecBench models to ``solar.api``."""

from __future__ import annotations

import shutil
from functools import partial
from pathlib import Path

from sol_execbench.core.platform.runtime import detect_rocm_device
from sol_execbench.core.solar_bridge.formal_device import (
    FORMAL_ARCHITECTURE,
    release_formal_device_memory,
    require_formal_device,
    solar_architecture_for_gfx_target,
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
    load_solar_workload_context,
    solar_conversion_request,
)
from solar.ir.contracts import DEFAULT_IR_PATH, IRPath, normalize_ir_path


def analyze_workload(
    *,
    problem_dir: str | Path,
    workload_uuid: str,
    output_dir: str | Path,
    device: str,
    orojenesis_home: str | Path | None,
    ir_path: IRPath | str = DEFAULT_IR_PATH,
    device_stage_lock_path: str | Path | None = None,
    device_stage_lock_timeout_seconds: float = 14_400.0,
) -> SolarAnalysisOutcome:
    """Adapt one workload and invoke SOLAR's benchmark-agnostic API."""
    require_formal_device(device)
    context = load_solar_workload_context(problem_dir, workload_uuid, device)
    return _invoke_solar(
        context=context,
        output_dir=Path(output_dir),
        device=device,
        orojenesis_home=orojenesis_home,
        ir_path=normalize_ir_path(ir_path),
        device_stage_lock_path=device_stage_lock_path,
        device_stage_lock_timeout_seconds=device_stage_lock_timeout_seconds,
    )


def analyze_workload_diagnostic(
    *,
    problem_dir: str | Path,
    workload_uuid: str,
    output_dir: str | Path,
    device: str,
    orojenesis_home: str | Path | None = None,
    ir_path: IRPath | str = DEFAULT_IR_PATH,
    device_stage_lock_path: str | Path | None = None,
    device_stage_lock_timeout_seconds: float = 14_400.0,
) -> SolarAnalysisOutcome:
    """Adapt one workload and invoke SOLAR's non-formal diagnostic path.

    Unlike :func:`analyze_workload`, this path never requires the formal
    gfx1200 device identity, selects the per-architecture profile from the
    detected gfx target, skips the verified-audit and Orojenesis gates, and
    returns a roofline bound even when the result is not publication-eligible.
    Results are engineering/inexact, never a formal SOLAR bound.
    """
    gfx_target = detect_rocm_device(device).gfx_target
    context = load_solar_workload_context(problem_dir, workload_uuid, device)
    return _invoke_solar(
        context=context,
        output_dir=Path(output_dir),
        device=device,
        orojenesis_home=orojenesis_home,
        ir_path=normalize_ir_path(ir_path),
        architecture=solar_architecture_for_gfx_target(gfx_target),
        formal=False,
        device_stage_lock_path=device_stage_lock_path,
        device_stage_lock_timeout_seconds=device_stage_lock_timeout_seconds,
    )


def _invoke_solar(
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
    from solar.api import (
        AnalysisExecutionPolicy,
        AnalysisFailure,
        AnalysisRequest,
        analyze,
    )

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
        execution_policy=AnalysisExecutionPolicy(
            device_stage_lock_path=(
                Path(device_stage_lock_path)
                if device_stage_lock_path is not None
                else None
            ),
            device_stage_lock_timeout_seconds=(
                device_stage_lock_timeout_seconds
            ),
            device_stage_cleanup=(
                partial(release_formal_device_memory, device)
                if device_stage_lock_path is not None
                else None
            ),
        ),
    )
    result = analyze(request)
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
        artifacts=tuple(artifact.to_dict() for artifact in result.artifacts),
        publication_eligible=result.publication_eligible,
    )
    if formal and not outcome.is_formal_publication:
        shutil.rmtree(result.output_dir, ignore_errors=True)
        return SolarAnalysisOutcome(
            status=SolarAnalysisStatus.FAILED,
            analysis_id=result.analysis_id,
            ir_path=request.ir_path,
            stage=SolarStage.FORMAL_ACCEPTANCE,
            reason_code="non_formal_bound",
            message="SOLAR formal bridge rejected a non-publication result",
        )
    return outcome

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
)
from sol_execbench.core.solar_bridge.models import (
    SolarAnalysisOutcome,
    SolarAnalysisStatus,
    SolarStage,
    SolarStageAuditOutcome,
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


def formal_producer_readiness() -> tuple[bool, str]:
    """Report whether this release can produce publication-grade SOLAR bounds."""
    from solar.api import formal_producer_readiness as solar_readiness

    readiness = solar_readiness()
    return readiness.ready, readiness.reason_code


def formal_architecture_profile_hash(
    architecture: str = FORMAL_ARCHITECTURE,
) -> str:
    """Return the canonical hash of a packaged SOLAR architecture profile."""
    from solar.api import architecture_profile_sha256

    return architecture_profile_sha256(architecture)


_ARCHITECTURE_BY_GFX_TARGET = {
    "gfx1200": FORMAL_ARCHITECTURE,
    "gfx942": "MI300X",
}


def _architecture_for_gfx_target(gfx_target: str) -> str:
    """Return the packaged SOLAR profile name for one gfx target."""
    architecture = _ARCHITECTURE_BY_GFX_TARGET.get(gfx_target)
    if architecture is None:
        raise ValueError(f"unsupported_solar_architecture:{gfx_target}")
    return architecture


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
        architecture=_architecture_for_gfx_target(gfx_target),
        formal=False,
        device_stage_lock_path=device_stage_lock_path,
        device_stage_lock_timeout_seconds=device_stage_lock_timeout_seconds,
    )


def audit_workload_stages(
    *,
    problem_dir: str | Path,
    workload_uuid: str,
    output_dir: str | Path,
    device: str,
    ir_path: IRPath | str = DEFAULT_IR_PATH,
) -> SolarStageAuditOutcome:
    """Run the exact extraction/conversion/replay gate for one corpus workload."""
    from solar.api import ConversionReadinessRequest, audit_conversion

    require_formal_device(device)
    selected_path = normalize_ir_path(ir_path)
    context = load_solar_workload_context(problem_dir, workload_uuid, device)
    result = audit_conversion(
        ConversionReadinessRequest(
            conversion=solar_conversion_request(
                context,
                device,
                selected_path,
            ),
            architecture=FORMAL_ARCHITECTURE,
            output_dir=Path(output_dir),
        ),
    )
    return SolarStageAuditOutcome.from_dict(result.to_dict())


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
        artifacts=tuple(artifact.__dict__ for artifact in result.artifacts),
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

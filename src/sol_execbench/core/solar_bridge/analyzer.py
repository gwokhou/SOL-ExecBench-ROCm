# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Workload entry points and readiness gates for the SOLAR bridge."""

from __future__ import annotations

from pathlib import Path

from sol_execbench.core.bench.gpu_execution import native_gpu_execution
from sol_execbench.core.platform.runtime import detect_rocm_device
from sol_execbench.core.solar_bridge.formal_device import (
    FORMAL_ARCHITECTURE,
    device_execution_policy,
    require_formal_device,
)
from sol_execbench.core.solar_bridge.invocation import invoke_solar
from sol_execbench.core.solar_bridge.models import (
    SolarAnalysisOutcome,
    SolarStageAuditOutcome,
)
from sol_execbench.core.solar_bridge.workload_context import (
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
    with native_gpu_execution(
        device, timeout_seconds=device_stage_lock_timeout_seconds
    ):
        require_formal_device(device)
        context = load_solar_workload_context(
            problem_dir, workload_uuid, device
        )
    return invoke_solar(
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
    with native_gpu_execution(
        device, timeout_seconds=device_stage_lock_timeout_seconds
    ):
        gfx_target = detect_rocm_device(device).gfx_target
        context = load_solar_workload_context(
            problem_dir, workload_uuid, device
        )
    return invoke_solar(
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

    with native_gpu_execution(device):
        require_formal_device(device)
        selected_path = normalize_ir_path(ir_path)
        context = load_solar_workload_context(
            problem_dir, workload_uuid, device
        )
        result = audit_conversion(
            ConversionReadinessRequest(
                conversion=solar_conversion_request(
                    context,
                    device,
                    selected_path,
                ),
                architecture=FORMAL_ARCHITECTURE,
                output_dir=Path(output_dir),
                execution_policy=device_execution_policy(device, None, 60.0),
            ),
        )
    return SolarStageAuditOutcome(
        status=result.status,
        analysis_id=result.analysis_id,
        ir_path=result.ir_path,
        output_dir=result.output_dir,
        architecture_sha256=result.architecture_sha256,
        stages=tuple(stage.to_dict() for stage in result.stages),
        artifacts=tuple(artifact.to_dict() for artifact in result.artifacts),
        failure_stage=result.failure_stage,
        reason_code=result.reason_code,
        message=result.message,
    )

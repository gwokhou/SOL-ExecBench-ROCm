# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""The sole in-process adapter from SOL ExecBench models to ``solar.api``."""

from __future__ import annotations

from pathlib import Path
import shutil

from sol_execbench.core.integrity import sha256_bytes
from sol_execbench.core.solar_bridge.models import (
    SolarAnalysisOutcome,
    SolarStageAuditOutcome,
    formal_precision_for_definition,
)
from sol_execbench.core.solar_bridge.workload_context import (
    SolarWorkloadContext,
    load_solar_workload_context,
)

FORMAL_ARCHITECTURE, FORMAL_GFX_TARGET = "RX_9060_XT", "gfx1200"


def formal_architecture_profile_hash(architecture: str = FORMAL_ARCHITECTURE) -> str:
    """Return the canonical SHA-256 of SOLAR's packaged architecture profile.

    This is the sole outer-package entry point that inspects a SOLAR profile's
    identity hash, so callers (including tests) never import ``solar`` directly.
    It intentionally wraps a private ``solar.api`` helper: the bridge owns the
    coupling to SOLAR internals so the rest of the benchmark stays decoupled.
    """
    from solar.api import _profile_hash
    from solar.rocm.architecture import ArchitectureProfile

    return _profile_hash(ArchitectureProfile.load(architecture))


def analyze_workload(
    *,
    problem_dir: str | Path,
    workload_uuid: str,
    output_dir: str | Path,
    device: str,
    orojenesis_home: str | Path | None,
) -> SolarAnalysisOutcome:
    """Adapt one workload and invoke SOLAR's benchmark-agnostic API."""
    _require_formal_device(device)
    context = load_solar_workload_context(problem_dir, workload_uuid, device)
    return _invoke_solar(
        context=context,
        output_dir=Path(output_dir),
        device=device,
        orojenesis_home=orojenesis_home,
    )


def audit_workload_stages(
    *,
    problem_dir: str | Path,
    workload_uuid: str,
    output_dir: str | Path,
    device: str,
) -> SolarStageAuditOutcome:
    """Run the exact extraction/conversion/replay gate for one corpus workload."""
    from solar.api import ConversionReadinessRequest, audit_conversion

    _require_formal_device(device)
    context = load_solar_workload_context(problem_dir, workload_uuid, device)
    definition, workload = context.definition, context.workload
    result = audit_conversion(
        ConversionReadinessRequest(
            analysis_id=f"{definition.name}:{workload.uuid}",
            reference=context.reference,
            input_factory=context.input_factory,
            reference_name=f"{definition.name}/definition.json#reference",
            reference_sha256=sha256_bytes(definition.reference.encode()),
            architecture=FORMAL_ARCHITECTURE,
            output_dir=Path(output_dir),
            device=device,
            atol=workload.tolerance.max_atol,
            rtol=workload.tolerance.max_rtol,
            required_matched_ratio=workload.tolerance.required_matched_ratio,
            max_error_cap=workload.tolerance.max_error_cap,
            allow_negative_inf=workload.tolerance.allow_negative_inf,
        )
    )
    return SolarStageAuditOutcome.from_dict(result.to_dict())


def _invoke_solar(
    *,
    context: SolarWorkloadContext,
    output_dir: Path,
    device: str,
    orojenesis_home: str | Path | None,
) -> SolarAnalysisOutcome:
    from solar.api import AnalysisFailure, AnalysisRequest, analyze

    definition, workload = context.definition, context.workload
    tolerance = workload.tolerance
    request = AnalysisRequest(
        analysis_id=f"{definition.name}:{workload.uuid}",
        reference=context.reference,
        input_factory=context.input_factory,
        reference_name=f"{definition.name}/definition.json#reference",
        reference_sha256=sha256_bytes(definition.reference.encode()),
        architecture=FORMAL_ARCHITECTURE,
        output_dir=output_dir,
        device=device,
        precision=formal_precision_for_definition(definition),
        require_orojenesis=True,
        orojenesis_home=orojenesis_home,
        atol=tolerance.max_atol,
        rtol=tolerance.max_rtol,
        required_matched_ratio=tolerance.required_matched_ratio,
        max_error_cap=tolerance.max_error_cap,
        allow_negative_inf=tolerance.allow_negative_inf,
    )
    result = analyze(request)
    if isinstance(result, AnalysisFailure):
        return SolarAnalysisOutcome(
            status=result.status,
            analysis_id=result.analysis_id,
            stage=result.stage,
            reason_code=result.reason_code,
            message=result.message,
        )
    outcome = SolarAnalysisOutcome(
        status=result.status,
        analysis_id=result.analysis_id,
        output_dir=str(result.output_dir),
        architecture_sha256=result.architecture_sha256,
        lower_bound_seconds=result.bound.seconds,
        bound_kind=result.bound.kind,
        limiting_resource=result.bound.limiting_resource,
        artifacts=tuple(artifact.__dict__ for artifact in result.artifacts),
        publication_eligible=result.publication_eligible,
    )
    if not outcome.is_formal_publication:
        shutil.rmtree(result.output_dir, ignore_errors=True)
        return SolarAnalysisOutcome(
            status="failed",
            analysis_id=result.analysis_id,
            stage="formal_acceptance",
            reason_code="non_formal_bound",
            message="SOLAR formal bridge rejected a non-publication result",
        )
    return outcome


def _require_formal_device(device: str) -> None:
    import torch

    if not torch.cuda.is_available() or not getattr(torch.version, "hip", None):
        raise RuntimeError("formal SOLAR analysis requires a ROCm device")
    selected = torch.device(device)
    index = (
        selected.index if selected.index is not None else torch.cuda.current_device()
    )
    properties = torch.cuda.get_device_properties(index)
    gfx_target = str(getattr(properties, "gcnArchName", "")).split(":", 1)[0]
    if gfx_target != FORMAL_GFX_TARGET:
        raise RuntimeError(
            f"formal SOLAR analysis requires {FORMAL_GFX_TARGET}, got {gfx_target or 'unknown'}; "
            "other AMD devices remain diagnostic evaluation targets"
        )

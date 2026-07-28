# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""The sole in-process adapter from SOL ExecBench models to ``solar.api``."""

from __future__ import annotations

import shutil
from pathlib import Path

from sol_execbench.core.integrity import sha256_bytes
from sol_execbench.core.solar_bridge.formal_device import (
    FORMAL_ARCHITECTURE,
)
from sol_execbench.core.solar_bridge.formal_device import (
    require_formal_device as _require_formal_device,
)
from sol_execbench.core.solar_bridge.models import (
    SolarAnalysisOutcome,
    SolarAnalysisStatus,
    SolarStage,
    SolarStageAuditOutcome,
    formal_precision_for_definition,
)
from sol_execbench.core.solar_bridge.workload_context import (
    SolarWorkloadContext,
    load_solar_workload_context,
)
from solar.routes import DEFAULT_ROUTE, Route


def formal_producer_readiness() -> tuple[bool, str]:
    """Report whether this release can produce publication-grade SOLAR bounds."""
    from solar.analysis.orojenesis import OROJENESIS_TRUSTED_MAPPER_SHA256

    if not OROJENESIS_TRUSTED_MAPPER_SHA256:
        return False, "formal_mapper_not_allowlisted"
    return True, "ready"


def formal_architecture_profile_hash(
    architecture: str = FORMAL_ARCHITECTURE,
) -> str:
    """Return the canonical hash of a packaged SOLAR architecture profile."""
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
    route: Route | str = DEFAULT_ROUTE,
) -> SolarAnalysisOutcome:
    """Adapt one workload and invoke SOLAR's benchmark-agnostic API."""
    _require_formal_device(device)
    context = load_solar_workload_context(problem_dir, workload_uuid, device)
    return _invoke_solar(
        context=context,
        output_dir=Path(output_dir),
        device=device,
        orojenesis_home=orojenesis_home,
        route=route,
    )


def audit_workload_stages(
    *,
    problem_dir: str | Path,
    workload_uuid: str,
    output_dir: str | Path,
    device: str,
    route: Route | str = DEFAULT_ROUTE,
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
            route=route,
            atol=workload.tolerance.max_atol,
            rtol=workload.tolerance.max_rtol,
            required_matched_ratio=workload.tolerance.required_matched_ratio,
            max_error_cap=workload.tolerance.max_error_cap,
            allow_negative_inf=workload.tolerance.allow_negative_inf,
        ),
    )
    return SolarStageAuditOutcome.from_dict(result.to_dict())


def _invoke_solar(
    *,
    context: SolarWorkloadContext,
    output_dir: Path,
    device: str,
    orojenesis_home: str | Path | None,
    route: Route | str = DEFAULT_ROUTE,
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
        route=route,
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
            status=SolarAnalysisStatus.FAILED,
            analysis_id=result.analysis_id,
            stage=SolarStage.FORMAL_ACCEPTANCE,
            reason_code="non_formal_bound",
            message="SOLAR formal bridge rejected a non-publication result",
        )
    return outcome

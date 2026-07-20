# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""The sole in-process adapter from SOL ExecBench models to ``solar.api``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from sol_execbench.core.bench.eval_runtime import load_reference_function
from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.integrity import sha256_bytes
from sol_execbench.core.solar_bridge.input_factory import build_input_factory
from sol_execbench.core.solar_bridge.models import (
    SolarAnalysisOutcome,
    formal_precision_for_definition,
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
    problem = Path(problem_dir).resolve()
    definition = Definition.model_validate_json(
        (problem / "definition.json").read_text()
    )
    workloads = _load_workloads(problem / "workload.jsonl")
    row_index, workload = _select_workload(workloads, workload_uuid)
    module, reference = load_reference_function(definition.reference)
    factory = build_input_factory(
        definition, workload, row_index, module, problem, device
    )
    return _invoke_solar(
        definition=definition,
        workload=workload,
        reference=reference,
        input_factory=factory,
        output_dir=Path(output_dir),
        device=device,
        orojenesis_home=orojenesis_home,
    )


def _invoke_solar(
    *,
    definition: Definition,
    workload: Workload,
    reference: Callable[..., Any],
    input_factory: Callable[[int], tuple[Any, ...]],
    output_dir: Path,
    device: str,
    orojenesis_home: str | Path | None,
) -> SolarAnalysisOutcome:
    from solar.api import AnalysisFailure, AnalysisRequest, analyze

    tolerance = workload.tolerance
    request = AnalysisRequest(
        analysis_id=f"{definition.name}:{workload.uuid}",
        reference=reference,
        input_factory=input_factory,
        reference_name=f"{definition.name}/definition.json#reference",
        reference_sha256=sha256_bytes(definition.reference.encode()),
        architecture=FORMAL_ARCHITECTURE,
        output_dir=output_dir,
        device=device,
        precision=formal_precision_for_definition(definition),
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
    return SolarAnalysisOutcome(
        status=result.status,
        analysis_id=result.analysis_id,
        output_dir=str(result.output_dir),
        architecture_sha256=result.architecture_sha256,
        lower_bound_seconds=result.bound.seconds,
        bound_kind=result.bound.kind,
        limiting_resource=result.bound.limiting_resource,
        artifacts=tuple(artifact.__dict__ for artifact in result.artifacts),
    )


def _load_workloads(path: Path) -> list[Workload]:
    return [
        Workload.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _select_workload(
    workloads: list[Workload], workload_uuid: str
) -> tuple[int, Workload]:
    matches = [
        (index, workload)
        for index, workload in enumerate(workloads)
        if workload.uuid == workload_uuid
    ]
    if len(matches) != 1:
        raise ValueError(f"workload UUID must match exactly once: {workload_uuid}")
    return matches[0]


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

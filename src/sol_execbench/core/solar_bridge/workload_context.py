# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Load one corpus workload into the sole SOLAR bridge."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from sol_execbench.core.bench.eval_runtime import load_reference_function
from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.workload import (
    CustomInput,
    SafetensorsInput,
    Workload,
    conservative_numeric_tolerance,
)
from sol_execbench.core.integrity import sha256_bytes
from sol_execbench.core.solar_bridge.input_factory import build_input_factory
from solar.ir.contracts import IRPath

if TYPE_CHECKING:
    from solar.api import ConversionRequest


@dataclass(frozen=True, slots=True, kw_only=True)
class SolarWorkloadContext:
    """Resolved reference and deterministic input factory for one workload."""

    definition: Definition
    workload: Workload
    reference: Callable[..., object]
    input_factory: Callable[[int], tuple[object, ...]]
    preserved_input_indices: tuple[int, ...] = ()


def load_solar_workload_context(
    problem_dir: str | Path,
    workload_uuid: str,
    device: str,
) -> SolarWorkloadContext:
    """Load and uniquely select a workload without importing SOLAR."""
    problem = Path(problem_dir).resolve()
    definition = Definition.model_validate_json(
        (problem / "definition.json").read_text(),
    )
    workloads = [
        Workload.model_validate_json(line)
        for line in (problem / "workload.jsonl").read_text().splitlines()
        if line.strip()
    ]
    row_index, workload = _select_workload(workloads, workload_uuid)
    module, reference = load_reference_function(definition.reference)
    factory = build_input_factory(
        definition,
        workload,
        row_index,
        module,
        problem,
        device,
    )
    preserved = structured_input_indices(definition, workload)
    return SolarWorkloadContext(
        definition=definition,
        workload=workload,
        reference=reference,
        input_factory=factory,
        preserved_input_indices=preserved,
    )


def structured_input_indices(
    definition: Definition,
    workload: Workload,
) -> tuple[int, ...]:
    """Return argument indices whose authored structure must remain unchanged."""
    return tuple(
        index
        for index, name in enumerate(definition.inputs)
        if isinstance(
            workload.inputs.get(name),
            (CustomInput, SafetensorsInput),
        )
    )


def solar_conversion_request(
    context: SolarWorkloadContext,
    device: str,
    ir_path: IRPath,
) -> ConversionRequest:
    """Build SOLAR's conversion request from one resolved workload context."""
    from solar.api import ConversionRequest, VerificationPolicy

    definition, workload = context.definition, context.workload
    tolerance = conservative_numeric_tolerance(workload.checks)
    return ConversionRequest(
        analysis_id=f"{definition.name}:{workload.uuid}",
        reference=context.reference,
        input_factory=context.input_factory,
        reference_name=f"{definition.name}/definition.json#reference",
        reference_sha256=sha256_bytes(definition.reference.encode()),
        ir_path=ir_path,
        verification=VerificationPolicy(
            atol=tolerance.max_atol,
            rtol=tolerance.max_rtol,
            required_matched_ratio=tolerance.required_matched_ratio,
            max_error_cap=tolerance.max_error_cap,
            allow_negative_inf=tolerance.allow_negative_inf,
            device=device,
            preserved_input_indices=context.preserved_input_indices,
        ),
    )


def _select_workload(
    workloads: list[Workload],
    workload_uuid: str,
) -> tuple[int, Workload]:
    matches = [
        (index, workload)
        for index, workload in enumerate(workloads)
        if workload.uuid == workload_uuid
    ]
    if len(matches) != 1:
        raise ValueError(
            f"workload UUID must match exactly once: {workload_uuid}",
        )
    return matches[0]


__all__ = [
    "SolarWorkloadContext",
    "load_solar_workload_context",
    "solar_conversion_request",
    "structured_input_indices",
]

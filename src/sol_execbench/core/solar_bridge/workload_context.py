# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Load one corpus workload into the sole SOLAR bridge."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from sol_execbench.core.bench.eval_runtime import load_reference_function
from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.solar_bridge.input_factory import build_input_factory


@dataclass(frozen=True, slots=True)
class SolarWorkloadContext:
    """Resolved reference and deterministic input factory for one workload."""

    definition: Definition
    workload: Workload
    reference: Callable[..., object]
    input_factory: Callable[[int], tuple[object, ...]]


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
    return SolarWorkloadContext(definition, workload, reference, factory)


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


__all__ = ["SolarWorkloadContext", "load_solar_workload_context"]

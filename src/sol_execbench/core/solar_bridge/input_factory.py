# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Benchmark input adaptation for the SOLAR process bridge."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from sol_execbench.core.bench.custom_inputs import isolated_torch_rng
from sol_execbench.core.bench.input_generation import gen_inputs
from sol_execbench.core.bench.safetensors_io import (
    FLASHINFER_TRACE_ENV,
    flashinfer_safetensors_env,
    load_safetensors,
)
from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.workload import Workload


def _safetensors_blob_roots(problem_dir: Path) -> list[Path]:
    resolved_problem = problem_dir.resolve()
    roots: list[Path] = [resolved_problem]
    problems_ancestor = next(
        (
            parent
            for parent in resolved_problem.parents
            if parent.name == "problems"
        ),
        None,
    )
    if problems_ancestor is not None:
        for parent in resolved_problem.parents:
            roots.append(parent)
            if parent == problems_ancestor:
                break
    trace_env = flashinfer_safetensors_env().get(FLASHINFER_TRACE_ENV)
    if trace_env:
        trace_root = Path(trace_env).resolve()
        if trace_root not in roots:
            roots.append(trace_root)
    return roots


def build_input_factory(
    definition: Definition,
    workload: Workload,
    row_index: int,
    reference_module: Any,
    problem_dir: Path,
    device: str,
) -> Callable[[int], tuple[Any, ...]]:
    """Bind benchmark inputs to SOLAR's deterministic seed factory contract."""
    custom = (
        getattr(reference_module, definition.custom_inputs_entrypoint)
        if definition.custom_inputs_entrypoint
        else None
    )
    safetensors = load_safetensors(
        definition,
        workload,
        blob_roots=_safetensors_blob_roots(problem_dir),
    )

    def factory(seed: int) -> tuple[Any, ...]:
        with isolated_torch_rng(seed):
            return tuple(
                gen_inputs(
                    definition,
                    workload,
                    device,
                    safe_tensors=safetensors,
                    custom_inputs_fn=custom,
                    row_index=row_index,
                    seed=seed,
                ),
            )

    return factory


__all__ = ["build_input_factory"]

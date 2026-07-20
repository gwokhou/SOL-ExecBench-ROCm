# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Benchmark input adaptation for the SOLAR process bridge."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from sol_execbench.core.bench.custom_inputs import isolated_torch_rng
from sol_execbench.core.bench.input_generation import gen_inputs
from sol_execbench.core.bench.safetensors_io import load_safetensors
from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.workload import Workload


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
        definition, workload, blob_roots=[problem_dir, *problem_dir.parents]
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
                )
            )

    return factory


__all__ = ["build_input_factory"]

"""CPU-only development input construction with invocation-specific seeds."""

from dataclasses import dataclass, field
from math import prod

import torch

from sol_execbench.core.bench.custom_inputs import derive_custom_input_seed
from sol_execbench.core.bench.judge.contracts import (
    MAX_DOT_PRODUCTS,
    MAX_INPUT_ELEMENTS,
    MAX_OUTPUT_ELEMENTS,
    JudgeCase,
    JudgeFamily,
    JudgeTask,
)
from sol_execbench.core.bench.judge.matrix_inputs import matrix_inputs
from sol_execbench.core.bench.judge.vector_inputs import vector_inputs
from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.workload import Workload


@dataclass(frozen=True, slots=True, kw_only=True)
class JudgeInputs:
    """Actual quantized tensors and explicit witness coordinates."""

    values: tuple[torch.Tensor, ...]
    witness_columns: tuple[int, ...] = ()
    construction: dict[str, tuple[int, ...]] = field(default_factory=dict)


def generate_development_inputs(
    case: JudgeCase,
    definition: Definition,
    workload: Workload,
    *,
    run_nonce: str,
    variation_index: int = 0,
) -> JudgeInputs:
    """Use the existing private-nonce seed contract without probing a device."""
    base_seed = case.require_development()
    if workload.uuid != case.workload_uuid:
        raise ValueError("judge workload identity mismatch")
    if len(run_nonce) != 64 or any(
        c not in "0123456789abcdef" for c in run_nonce
    ):
        raise ValueError("judge nonce must be 32 lowercase hexadecimal bytes")
    if variation_index < 0:
        raise ValueError("variation index must be nonnegative")
    axes = definition.get_resolved_axes_values(workload.axes)
    _bound_inputs(case.task, axes)
    seed = derive_custom_input_seed(
        definition,
        workload,
        base_seed=base_seed,
        round_index=0,
        run_nonce=run_nonce,
        variation_index=variation_index,
    )
    return generate_seeded_inputs(case.task, case.family, axes, seed)


def generate_seeded_inputs(
    task: JudgeTask, family: JudgeFamily, axes: dict[str, int], seed: int
) -> JudgeInputs:
    """Build bounded quantized inputs from a trusted invocation seed."""
    _bound_inputs(task, axes)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if task is JudgeTask.GEMM:
        tensors, columns, construction = matrix_inputs(family, axes, generator)
        return JudgeInputs(
            values=tensors, witness_columns=columns, construction=construction
        )
    values, construction = vector_inputs(family, axes, generator)
    return JudgeInputs(values=values, construction=construction)


def _bound_inputs(task: JudgeTask, axes: dict[str, int]) -> None:
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in axes.values()
    ):
        raise ValueError("judge axes must be positive")
    if task is JudgeTask.GEMM:
        size = axes["M"] * axes["K"] + axes["K"] * axes["N"]
        if (
            axes["M"] * axes["N"] > MAX_OUTPUT_ELEMENTS
            or axes["M"] * axes["N"] * axes["K"] > MAX_DOT_PRODUCTS
        ):
            raise ValueError("judge output or contraction budget exceeded")
    elif task is JudgeTask.SUM:
        size = prod((axes["M"], axes["N"]))
    else:
        if axes["B"] * axes["N"] > MAX_OUTPUT_ELEMENTS:
            raise ValueError("judge output budget exceeded")
        size = axes["B"] * axes["N"] + 2 * axes["N"]
    if size > MAX_INPUT_ELEMENTS:
        raise ValueError("judge input element budget exceeded")

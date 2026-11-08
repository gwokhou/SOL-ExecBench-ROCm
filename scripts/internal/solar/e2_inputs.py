"""Trusted E2 input adapter; no candidate or oracle imports."""

import torch

from sol_execbench.core.bench.judge.contracts import JudgeFamily, JudgeTask
from sol_execbench.core.bench.judge.inputs import generate_seeded_inputs


def factory(
    task: str,
    family: str,
    axes_and_scalars: dict[str, int | float | bool],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Use the frozen generator with the trusted invocation seed, once."""
    selected = JudgeTask(task)
    axes = {
        key: int(value)
        for key, value in axes_and_scalars.items()
        if key != "eps"
    }
    generated = generate_seeded_inputs(
        selected, JudgeFamily(family), axes, torch.initial_seed()
    )
    names = {
        JudgeTask.GEMM: ("A", "B"),
        JudgeTask.SUM: ("x",),
        JudgeTask.LAYERNORM: ("x", "weight", "bias"),
    }[selected]
    return {
        name: value.to(device)
        for name, value in zip(
            names, generated.values[: len(names)], strict=True
        )
    }

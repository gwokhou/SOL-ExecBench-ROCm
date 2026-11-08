"""Frozen reduction and LayerNorm development families."""

import torch

from sol_execbench.core.bench.judge.contracts import SMALL_VALUE, JudgeFamily

_POSITIVE_GRID = 32
_REDUCTION_MAGNITUDE = 4096
_NORM_OFFSET = 4096
_NORM_VARIATION = 2.0**-8
_NORM_EPSILON = 1e-5
_AFFINE_WEIGHT = (0.5, 2.0, -1.0, 1.5)
_AFFINE_BIAS = (-0.5, 0.25, 1.0, -0.25)


def vector_inputs(
    family: JudgeFamily,
    axes: dict[str, int],
    generator: torch.Generator,
) -> tuple[tuple[torch.Tensor, ...], dict[str, tuple[int, ...]]]:
    """Produce CPU inputs, retaining the nontrivial affine witness row."""
    n = axes["N"]
    if n % 2:
        raise ValueError("vector witnesses require even N")
    if family.value.startswith("R"):
        value, construction = _reduction(family, axes["M"], n, generator)
        return (value,), construction
    batch = axes["B"]
    alternating = 1 - 2 * (torch.arange(n) % 2)
    weight, bias = (
        torch.ones(n, dtype=torch.float32),
        torch.zeros(n, dtype=torch.float32),
    )
    if family is JudgeFamily.L0:
        row = _NORM_OFFSET + alternating.double() * _NORM_VARIATION
    elif family is JudgeFamily.L1:
        row = 1 + alternating.double() * SMALL_VALUE
    elif family is JudgeFamily.L2:
        row = alternating.double() * _NORM_VARIATION
    elif family is JudgeFamily.L3:
        x = torch.randint(-16, 17, (batch, n), generator=generator).float() / 16
        x[0] = 0
        index = torch.arange(n) % len(_AFFINE_WEIGHT)
        weight = torch.tensor(_AFFINE_WEIGHT, dtype=torch.float32)[index]
        bias = torch.tensor(_AFFINE_BIAS, dtype=torch.float32)[index]
        order = torch.randperm(n, generator=generator)
        return (
            x[:, order],
            weight[order],
            bias[order],
            torch.tensor(_NORM_EPSILON, dtype=torch.float32),
        ), {"feature_permutation": tuple(order.tolist())}
    else:
        raise ValueError("not a vector input family")
    order = torch.randperm(n, generator=generator)
    x = row[order].float().repeat(batch, 1)
    return (
        x,
        weight,
        bias,
        torch.tensor(_NORM_EPSILON, dtype=torch.float32),
    ), {"feature_permutation": tuple(order.tolist())}


def _reduction(
    family: JudgeFamily,
    rows: int,
    n: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, tuple[int, ...]]]:
    if family is JudgeFamily.R0:
        x = (
            1
            + torch.randint(
                0, _POSITIVE_GRID, (rows, n), generator=generator
            ).float()
            / _POSITIVE_GRID
        )
        x[:, -1] = 1
        return x, {}
    if family is JudgeFamily.R2:
        row = torch.ones(n, dtype=torch.float32)
        row[1::2] = SMALL_VALUE
        order = torch.randperm(n, generator=generator)
    else:
        row = torch.zeros(n, dtype=torch.float32)
        triples = n // 3
        row[: triples * 3] = torch.tensor(
            (_REDUCTION_MAGNITUDE, 1, -_REDUCTION_MAGNITUDE),
            dtype=torch.float32,
        ).repeat(triples)
        if family is JudgeFamily.R3:
            order = torch.randperm(n, generator=generator)
        elif family is JudgeFamily.R1:
            # Row signs vary per invocation without altering the cancellation.
            signs = torch.randint(0, 2, (rows, 1), generator=generator) * 2 - 1
            return row.repeat(rows, 1) * signs, {
                "row_signs": tuple(signs.reshape(-1).tolist())
            }
        else:
            raise ValueError("not a reduction input family")
    return row[order].repeat(rows, 1), {
        "feature_permutation": tuple(order.tolist())
    }

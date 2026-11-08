"""Structured BF16 matrix inputs; all randomness stays on the CPU."""

import torch

from sol_execbench.core.bench.judge.contracts import SMALL_VALUE, JudgeFamily

_GRID_LOW = -16
_GRID_HIGH = 17
_GRID_SCALE = 16
_SCALE_MIN = -8
_SCALE_PERIOD = 17
_CANCELLATION_MAGNITUDE = 256
_RESIDUAL = 1 / 16


def _grid(shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
    return (
        torch.randint(
            _GRID_LOW,
            _GRID_HIGH,
            shape,
            generator=generator,
            device="cpu",
        ).double()
        / _GRID_SCALE
    )


def matrix_inputs(
    family: JudgeFamily,
    axes: dict[str, int],
    generator: torch.Generator,
) -> tuple[
    tuple[torch.Tensor, ...], tuple[int, ...], dict[str, tuple[int, ...]]
]:
    """Construct frozen witnesses before casting to the actual BF16 inputs."""
    m, k, n = axes["M"], axes["K"], axes["N"]
    if k < 4 or k % 2 or n < 2:
        raise ValueError("matrix witness requires even K >= 4 and N >= 2")
    columns: tuple[int, ...] = ()
    construction: dict[str, tuple[int, ...]] = {}
    if family is JudgeFamily.G2:
        a, b, columns = _small_nonzero(m, k, n, generator)
    elif family is JudgeFamily.G3:
        a = torch.ones((m, k), dtype=torch.float64)
        b = torch.zeros((k, n), dtype=torch.float64)
        b[: k - 2 : 2] = _CANCELLATION_MAGNITUDE
        b[1 : k - 2 : 2] = -_CANCELLATION_MAGNITUDE
        b[k - 2] = _RESIDUAL
        order = torch.randperm(k, generator=generator)
        signs = torch.randint(0, 2, (n,), generator=generator) * 2 - 1
        a, b = a[:, order], b[order] * signs
        construction = {
            "inner_permutation": tuple(order.tolist()),
            "column_signs": tuple(signs.tolist()),
        }
    else:
        a, b = _grid((m, k), generator), _grid((k, n), generator)
        a[0] = 0
        b[:, 0] = 0
        a[0, 0] = b[0, 0] = 1
        if family is JudgeFamily.G1:
            exponents = torch.arange(k) % _SCALE_PERIOD + _SCALE_MIN
            exponents = exponents[torch.randperm(k, generator=generator)]
            construction["scale_exponents"] = tuple(exponents.tolist())
            scale = torch.exp2(exponents.double())
            a, b = a * scale, b / scale[:, None]
        elif family is not JudgeFamily.G0:
            raise ValueError("not a matrix input family")
    return (a.to(torch.bfloat16), b.to(torch.bfloat16)), columns, construction


def _small_nonzero(
    m: int,
    k: int,
    n: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    a = torch.zeros((m, k), dtype=torch.float64)
    b = torch.zeros((k, n), dtype=torch.float64)
    a[:, : k // 2] = _grid((m, k // 2), generator)
    b[k // 2 :] = _grid((k // 2, n), generator)
    b[:, :2] = 0
    b[k - 2 :] = 0
    a[:, k - 1] = a[:, k - 2] = 1
    b[k - 1, 0] = 1
    b[k - 2, 1] = SMALL_VALUE
    order = torch.randperm(n, generator=generator)
    inverse = torch.argsort(order)
    columns = (int(inverse[0]), int(inverse[1]))
    return a, b[:, order], columns

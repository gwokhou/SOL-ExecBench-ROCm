"""Independent Decimal checks at positions chosen before inspecting outputs."""

from decimal import Decimal, localcontext
from fractions import Fraction

import torch

from sol_execbench.core.bench.judge.contracts import JudgeTask
from sol_execbench.core.bench.judge.inputs import JudgeInputs

_CHECK_PRECISION = 120


def certify_oracle(
    task: JudgeTask,
    inputs: JudgeInputs,
    rounded: torch.Tensor,
    exact: tuple[Fraction, ...] | tuple[tuple[str, str], ...],
) -> int:
    """Check fixed corner/midpoint and declared witness columns independently."""
    rows = sorted({0, rounded.shape[0] // 2, rounded.shape[0] - 1})
    columns = sorted(
        {
            0,
            rounded.shape[1] // 2,
            rounded.shape[1] - 1,
            *inputs.witness_columns,
        }
    )
    count = 0
    with localcontext() as context:
        context.prec = _CHECK_PRECISION
        for row in rows:
            for column in columns:
                result = _decimal_output(task, inputs.values, row, column)
                witness = exact[row * rounded.shape[1] + column]
                if isinstance(witness, Fraction):
                    if Fraction(result) != witness:
                        raise ValueError(
                            "independent Decimal/integer oracle disagreement"
                        )
                elif not Decimal(witness[0]) <= result <= Decimal(witness[1]):
                    raise ValueError(
                        "independent Decimal result outside certified interval"
                    )
                count += 1
    return count


def _decimal_output(
    task: JudgeTask, values: tuple[torch.Tensor, ...], row: int, column: int
) -> Decimal:
    if task is JudgeTask.GEMM:
        return sum(
            (
                Decimal(float(a)) * Decimal(float(b))
                for a, b in zip(
                    values[0][row], values[1][:, column], strict=True
                )
            ),
            Decimal(0),
        )
    x = [Decimal(float(value)) for value in values[0][row]]
    if task is JudgeTask.SUM:
        return sum(x, Decimal(0))
    mean = sum(x, Decimal(0)) / len(x)
    variance = sum(((value - mean) ** 2 for value in x), Decimal(0)) / len(x)
    scaled = (x[column] - mean) / (variance + Decimal(float(values[3]))).sqrt()
    return scaled * Decimal(float(values[1][column])) + Decimal(
        float(values[2][column])
    )

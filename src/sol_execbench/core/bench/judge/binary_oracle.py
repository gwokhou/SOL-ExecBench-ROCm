"""Exact dyadic arithmetic and correctly rounded binary outputs."""

from fractions import Fraction
from math import ldexp

import numpy as np
import torch

from sol_execbench.core.bench.judge.contracts import (
    MAX_DOT_PRODUCTS,
    MAX_INPUT_ELEMENTS,
    MAX_OUTPUT_ELEMENTS,
    JudgeTask,
)

_FLOAT_PRECISION = {torch.float32: 24, torch.bfloat16: 8, torch.float64: 53}
_MIN_EXPONENT = {
    torch.float32: -126,
    torch.bfloat16: -126,
    torch.float64: -1022,
}
_MAX_EXPONENT = {torch.float32: 127, torch.bfloat16: 127, torch.float64: 1023}
_INT64_MAX = (1 << 63) - 1


def round_fraction(value: Fraction, dtype: torch.dtype) -> float:
    """Round a rational directly to the target format, avoiding double rounding."""
    if dtype not in _FLOAT_PRECISION:
        raise ValueError("unsupported oracle output dtype")
    if not value:
        return 0.0
    sign = -1 if value < 0 else 1
    numerator, denominator = abs(value.numerator), value.denominator
    exponent = numerator.bit_length() - denominator.bit_length()
    if exponent >= 0:
        exponent -= numerator < (denominator << exponent)
    else:
        exponent -= (numerator << -exponent) < denominator
    precision = _FLOAT_PRECISION[dtype]
    step = max(exponent, _MIN_EXPONENT[dtype]) - precision + 1
    if step >= 0:
        denominator <<= step
    else:
        numerator <<= -step
    quotient, remainder = divmod(numerator, denominator)
    if 2 * remainder > denominator or (
        2 * remainder == denominator and quotient % 2
    ):
        quotient += 1
    if quotient and quotient.bit_length() - 1 + step > _MAX_EXPONENT[dtype]:
        raise ValueError("oracle result overflows declared output dtype")
    return sign * ldexp(float(quotient), step)


def require_cpu_inputs(values: tuple[torch.Tensor, ...]) -> None:
    """Bound actual work before copying or allocating oracle buffers."""
    if not values or any(value.numel() == 0 for value in values):
        raise ValueError("oracle requires nonempty inputs")
    if sum(value.numel() for value in values) > MAX_INPUT_ELEMENTS:
        raise ValueError("oracle input element budget exceeded")
    for value in values:
        if value.device.type != "cpu":
            raise ValueError("offline oracle requires CPU tensors")
        if value.dtype not in _FLOAT_PRECISION:
            raise ValueError("oracle requires a declared floating input dtype")
        if not bool(torch.isfinite(value).all()):
            raise ValueError("oracle input is not finite")


def _integers(value: torch.Tensor) -> tuple[np.ndarray, int, int]:
    array = value.detach().double().numpy()
    ratios = [float(item).as_integer_ratio() for item in array.flat]
    shift = max((den.bit_length() - 1 for _, den in ratios), default=0)
    integers = [num << (shift - den.bit_length() + 1) for num, den in ratios]
    maximum = max(map(abs, integers), default=0)
    dtype = np.int64 if maximum <= _INT64_MAX else object
    return np.array(integers, dtype=dtype).reshape(array.shape), shift, maximum


def exact_dyadic_output(
    task: JudgeTask,
    values: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, tuple[Fraction, ...]]:
    """Compute every output with proved-safe integers or arbitrary precision."""
    require_cpu_inputs(values)
    first = values[0]
    if first.ndim != 2:
        raise ValueError("dyadic oracle expects rank-two inputs")
    output_count = first.shape[0]
    if task is JudgeTask.GEMM:
        if (
            len(values) != 2
            or values[1].ndim != 2
            or first.shape[1] != values[1].shape[0]
        ):
            raise ValueError("oracle matmul dimensions mismatch")
        output_count *= values[1].shape[1]
        if output_count * first.shape[1] > MAX_DOT_PRODUCTS:
            raise ValueError("oracle contraction work budget exceeded")
    elif task is not JudgeTask.SUM or len(values) != 1:
        raise ValueError("not a dyadic oracle operation")
    if output_count > MAX_OUTPUT_ELEMENTS:
        raise ValueError("oracle output element budget exceeded")
    left, shift, maximum = _integers(first)
    if task is JudgeTask.GEMM:
        right, right_shift, right_maximum = _integers(values[-1])
        bound = maximum * right_maximum * first.shape[1]
        if bound > _INT64_MAX:
            left, right = left.astype(object), right.astype(object)
        raw = left @ right
        shift += right_shift
    else:
        if maximum * first.shape[1] > _INT64_MAX:
            left = left.astype(object)
        raw = left.sum(axis=-1, keepdims=True)
    exact = tuple(Fraction(int(item), 1 << shift) for item in raw.flat)
    rounded = torch.tensor(
        [round_fraction(item, first.dtype) for item in exact],
        dtype=first.dtype,
    ).reshape(raw.shape)
    return rounded, exact

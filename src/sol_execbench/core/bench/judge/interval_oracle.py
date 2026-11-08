"""Directed decimal intervals for the preregistered LayerNorm oracle."""

from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, Context, Decimal
from fractions import Fraction
from functools import lru_cache

import torch

from sol_execbench.core.bench.judge.binary_oracle import (
    require_cpu_inputs,
    round_fraction,
)
from sol_execbench.core.bench.judge.contracts import MAX_OUTPUT_ELEMENTS

ORACLE_PRECISIONS = (80, 160)


class OracleUncertainError(ValueError):
    """The frozen precision budget cannot identify a unique FP32 result."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Interval:
    """Closed decimal enclosure, with directed arithmetic at both endpoints."""

    lower: Decimal
    upper: Decimal


@dataclass(frozen=True, slots=True, kw_only=True)
class IntervalArithmetic:
    """One independent precision context, without global decimal mutation."""

    precision: int

    @property
    def floor(self) -> Context:
        """Return a local lower-directed context."""
        return Context(prec=self.precision, rounding=ROUND_FLOOR)

    @property
    def ceiling(self) -> Context:
        """Return a local upper-directed context."""
        return Context(prec=self.precision, rounding=ROUND_CEILING)

    def rational(self, value: Fraction) -> Interval:
        """Enclose an exact rational."""
        n, d = Decimal(value.numerator), Decimal(value.denominator)
        return Interval(
            lower=self.floor.divide(n, d), upper=self.ceiling.divide(n, d)
        )

    def add(self, a: Interval, b: Interval) -> Interval:
        """Enclose addition."""
        return Interval(
            lower=self.floor.add(a.lower, b.lower),
            upper=self.ceiling.add(a.upper, b.upper),
        )

    def multiply(self, a: Interval, b: Interval) -> Interval:
        """Enclose products of signed intervals."""
        pairs = tuple(
            (x, y) for x in (a.lower, a.upper) for y in (b.lower, b.upper)
        )
        return Interval(
            lower=min(self.floor.multiply(x, y) for x, y in pairs),
            upper=max(self.ceiling.multiply(x, y) for x, y in pairs),
        )

    def inverse_sqrt(self, value: Fraction) -> Interval:
        """Enclose inverse sqrt; Decimal sqrt itself rounds to nearest."""
        enclosed = self.rational(value)
        if enclosed.lower <= 0:
            raise ValueError("LayerNorm variance plus epsilon must be positive")
        low = self.floor.sqrt(enclosed.lower).next_minus(self.floor)
        high = self.ceiling.sqrt(enclosed.upper).next_plus(self.ceiling)
        return Interval(
            lower=self.floor.divide(Decimal(1), high),
            upper=self.ceiling.divide(Decimal(1), low),
        )


@lru_cache(maxsize=32)
def _row_oracle(
    row: tuple[float, ...],
    weights: tuple[float, ...],
    biases: tuple[float, ...],
    epsilon: float,
) -> tuple[tuple[float, ...], tuple[tuple[str, str], ...], int]:
    values = tuple(Fraction(value) for value in row)
    mean = sum(values, Fraction()) / len(values)
    variance = sum(((value - mean) ** 2 for value in values), Fraction()) / len(
        values
    )
    for precision in ORACLE_PRECISIONS:
        arithmetic = IntervalArithmetic(precision=precision)
        inverse = arithmetic.inverse_sqrt(variance + Fraction(epsilon))
        outputs: list[float] = []
        bounds: list[tuple[str, str]] = []
        for value, weight, bias in zip(values, weights, biases, strict=True):
            centered = arithmetic.rational((value - mean) * Fraction(weight))
            result = arithmetic.add(
                arithmetic.multiply(centered, inverse),
                arithmetic.rational(Fraction(bias)),
            )
            lower = round_fraction(Fraction(result.lower), torch.float32)
            upper = round_fraction(Fraction(result.upper), torch.float32)
            if lower != upper:
                break
            outputs.append(lower)
            bounds.append((str(result.lower), str(result.upper)))
        if len(outputs) == len(values):
            return tuple(outputs), tuple(bounds), precision
    raise OracleUncertainError(
        "FP32 rounding remains ambiguous at 160 decimal digits"
    )


def layernorm_oracle(
    values: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, tuple[tuple[str, str], ...], int]:
    """Certify the entire FP32 output without using the tested reference."""
    require_cpu_inputs(values)
    if len(values) != 4:
        raise ValueError("LayerNorm oracle requires x, weight, bias, epsilon")
    x, weight, bias, epsilon = values
    if (
        x.ndim != 2
        or weight.shape != x.shape[-1:]
        or bias.shape != weight.shape
    ):
        raise ValueError("LayerNorm oracle shapes differ")
    if x.numel() > MAX_OUTPUT_ELEMENTS or epsilon.numel() != 1:
        raise ValueError(
            "LayerNorm oracle output budget or epsilon shape invalid"
        )
    weights, biases = (
        tuple(weight.double().tolist()),
        tuple(bias.double().tolist()),
    )
    rows: list[tuple[float, ...]] = []
    bounds: list[tuple[str, str]] = []
    precision = ORACLE_PRECISIONS[0]
    for row in x.double().tolist():
        result, row_bounds, used = _row_oracle(
            tuple(row), weights, biases, float(epsilon)
        )
        rows.append(result)
        bounds.extend(row_bounds)
        precision = max(precision, used)
    return torch.tensor(rows, dtype=torch.float32), tuple(bounds), precision

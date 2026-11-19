"""Check the E2 reference error envelope against known strict boundaries."""

from fractions import Fraction

import numpy as np
import pytest

from scripts.internal.solar.e2_reference_margin import reference_margin


@pytest.mark.parametrize(
    ("actual", "reference", "oracle", "atol", "rtol", "sign"),
    (
        ([1.0], [1.0], [1.0], 0.125, 0.125, "positive"),
        ([1.5], [1.0], [1.0], 0.125, 0.125, "negative"),
        ([1.25], [1.0], [1.0], 0.125, 0.125, "ambiguous"),
        ([1.25], [1.125], [1.0], 0.125, 0.125, "ambiguous"),
        ([1.0, -4.0], [1.0, -1.125], [1.0, -1.0], 0.125, 0.125, "negative"),
        ([0.0, 0.0], [0.0, 0.0], [0.0, 0.0], 0.0, 0.0, "ambiguous"),
    ),
    ids=(
        "exact-reference-pass",
        "exact-reference-fail",
        "exact-contract-boundary",
        "observed-reference-error-spans-decision",
        "one-failing-element-suffices",
        "zero-tolerance-needs-strict-sign",
    ),
)
def test_reference_margin(actual, reference, oracle, atol, rtol, sign):
    actual, reference, oracle = (
        np.array(values, dtype=np.float64)
        for values in (actual, reference, oracle)
    )
    result = reference_margin(actual, reference, oracle, atol=atol, rtol=rtol)
    if sign == "positive":
        assert result.lower > 0
    elif sign == "negative":
        assert result.upper < 0
    else:
        assert result.lower <= 0 <= result.upper

    # Check exact margins at interval endpoints and internal breakpoints.
    minima, maxima = [], []
    for a, r, q in zip(actual, reference, oracle, strict=True):
        act, ref, center = map(Fraction, (float(a), float(r), float(q)))
        radius = abs(ref - center)
        left, right = center - radius, center + radius
        values = [left, right] + [x for x in (act, 0) if left <= x <= right]
        margins = [
            Fraction(atol) + Fraction(rtol) * abs(x) - abs(act - x)
            for x in values
        ]
        minima.append(min(margins))
        maxima.append(max(margins))
    assert Fraction(result.lower) <= min(minima)
    assert min(maxima) <= Fraction(result.upper)

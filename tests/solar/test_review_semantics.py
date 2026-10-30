"""Precision and gradient regressions from the full-repository review."""

from typing import cast

import pytest
import torch

from solar.errors import GradientVerificationError
from solar.verification.contracts import VerificationPolicy
from solar.verification.errors import VerificationError
from solar.verification.executor import IRGraphExecutor
from solar.verification.gradients import verify_gradients
from solar.verification.numerics import assert_close
from solar.verification.policy import verification_policy_record
from solar.verification.verify import _validated_recorded_tolerance


@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_double_precision_error_is_not_rounded_away(dtype):
    expected = torch.tensor([1.0], dtype=dtype)
    actual = torch.tensor([1.0 + 2.0**-30], dtype=dtype)
    with pytest.raises(VerificationError):
        assert_close(actual, expected, 0.0, 0.0, max_error_cap=0.0)
    assert_close(actual, expected, 2.0**-29, 0.0)


@pytest.mark.parametrize("reverse_indices", [False, True])
def test_gradient_order_matches_requested_inputs(reverse_indices):
    def reference(x, y):
        return x + 2 * y

    indices = (1, 0) if reverse_indices else (0, 1)
    result = verify_gradients(
        reference,
        cast("IRGraphExecutor", reference),
        {"source_input_indices": [0, 1]},
        (torch.ones(2), torch.ones(2)),
        VerificationPolicy(atol=0, rtol=0, gradient_input_indices=indices),
    )
    assert result["gradient_inputs_verified"] == 2


def test_opposite_output_gradients_cannot_cancel():
    def reference(x):
        return x.clone(), -x.clone()

    def wrong(x):
        return 2 * x - x.detach(), -2 * x + x.detach()

    with pytest.raises(GradientVerificationError):
        verify_gradients(
            reference,
            cast("IRGraphExecutor", wrong),
            {"source_input_indices": [0]},
            (torch.tensor([0.3, -0.7]),),
            VerificationPolicy(atol=0, rtol=0),
        )


def test_no_grad_output_pattern_is_preserved():
    def reference(x):
        with torch.no_grad():
            a, b = x * 2, -x * 2
        return x.clone(), a, b

    def wrong(x):
        return x.clone(), x * 2, -x * 2

    policy = VerificationPolicy(atol=0, rtol=0)
    args = ({"source_input_indices": [0]}, (torch.ones(2),), policy)
    assert (
        verify_gradients(reference, cast("IRGraphExecutor", reference), *args)[
            "gradient_inputs_verified"
        ]
        == 1
    )
    with pytest.raises(GradientVerificationError, match="differentiability"):
        verify_gradients(reference, cast("IRGraphExecutor", wrong), *args)


@pytest.mark.parametrize("field", ["gradient_atol", "gradient_rtol"])
@pytest.mark.parametrize("explicit", [True, False])
def test_replay_enforces_effective_gradient_tolerance(field, explicit):
    required = VerificationPolicy(
        atol=0, rtol=0, **({field: 0} if explicit else {})
    )
    weak = VerificationPolicy(
        atol=0,
        rtol=0,
        gradient_atol=1 if field == "gradient_atol" else None,
        gradient_rtol=1 if field == "gradient_rtol" else None,
    )
    with pytest.raises(VerificationError, match="weaker"):
        _validated_recorded_tolerance(
            verification_policy_record(weak), required
        )
    assert (
        _validated_recorded_tolerance(
            verification_policy_record(required), required
        )
        == required
    )


@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2],
)
def test_low_precision_comparison_still_promotes_without_float8_type_promotion(
    dtype,
):
    value = torch.ones(2).to(dtype)
    assert_close(value, value, 0, 0)


def test_complex_component_vjps_cannot_cancel():
    def reference(x):
        return torch.complex(x, x)

    def wrong(x):
        return torch.complex(2 * x - x.detach(), x.detach())

    with pytest.raises(GradientVerificationError):
        verify_gradients(
            reference,
            cast("IRGraphExecutor", wrong),
            {"source_input_indices": [0]},
            (torch.ones(2),),
            VerificationPolicy(atol=0, rtol=0),
        )


def test_recorded_gradient_fallback_cannot_weaken_explicit_request():
    recorded = VerificationPolicy(atol=1, rtol=1)
    required = VerificationPolicy(
        atol=1, rtol=1, gradient_atol=0, gradient_rtol=0
    )
    with pytest.raises(VerificationError, match="weaker"):
        _validated_recorded_tolerance(
            verification_policy_record(recorded), required
        )

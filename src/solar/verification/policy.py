# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Canonical policy and case records shared by verification and proof reuse."""

from collections.abc import Mapping

from solar.types import DynamicValue
from solar.verification.contracts import TolerancePolicy, VerificationPolicy
from solar.verification.errors import VerificationError

MAX_VERIFICATION_CASES = 1024


def verification_cases(
    parameters: Mapping[str, DynamicValue],
    policy: VerificationPolicy,
) -> list[dict[str, DynamicValue]]:
    """Construct the exact bounded seed-pattern product required by a proof."""
    if not policy.patterns:
        raise VerificationError(
            "trusted verification requires boundary patterns"
        )
    if len(policy.seeds) * len(policy.patterns) > MAX_VERIFICATION_CASES:
        raise VerificationError("verification cases exceed the case budget")
    if len({int(seed) for seed in policy.seeds}) < 3:
        raise VerificationError(
            "trusted verification requires at least three seeds",
        )
    if not {"random", "zeros", "boundary"}.issubset(policy.patterns):
        raise VerificationError(
            "trusted verification requires random, zeros, and boundary patterns",
        )
    return [
        {
            "parameters": dict(parameters),
            "seed": int(seed),
            "pattern": str(pattern),
        }
        for seed in policy.seeds
        for pattern in policy.patterns
    ]


def _tolerance_record(policy: TolerancePolicy) -> dict[str, DynamicValue]:
    return {
        "atol": float(policy.atol),
        "rtol": float(policy.rtol),
        "required_matched_ratio": float(policy.required_matched_ratio),
        "max_error_cap": policy.max_error_cap,
        "allow_negative_inf": bool(policy.allow_negative_inf),
    }


def verification_policy_record(
    policy: VerificationPolicy,
) -> dict[str, DynamicValue]:
    """Serialize all numerical, preserved-input and gradient policy choices."""
    return {
        **_tolerance_record(policy),
        "preserved_input_indices": list(policy.preserved_input_indices),
        "verify_gradients": policy.verify_gradients,
        "gradient_input_indices": (
            list(policy.gradient_input_indices)
            if policy.gradient_input_indices is not None
            else None
        ),
        "gradient_atol": policy.gradient_atol,
        "gradient_rtol": policy.gradient_rtol,
    }

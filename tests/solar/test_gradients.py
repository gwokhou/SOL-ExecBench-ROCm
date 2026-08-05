# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import cast

import pytest
import torch

from solar.errors import GradientVerificationError
from solar.verification.contracts import VerificationPolicy
from solar.verification.executor import IRGraphExecutor
from solar.verification.gradients import verify_gradients


def _policy() -> VerificationPolicy:
    return VerificationPolicy(atol=0.0, rtol=0.0, device="cpu")


def _nan_gradient(value: torch.Tensor) -> torch.Tensor:
    return value / value


def test_gradient_verification_accepts_only_positionally_matching_nan() -> None:
    inputs = (torch.zeros(4),)
    graph = {"source_input_indices": [0]}

    stats = verify_gradients(
        _nan_gradient,
        cast(IRGraphExecutor, _nan_gradient),
        graph,
        inputs,
        _policy(),
    )

    assert stats == {"gradient_inputs_verified": 1.0}


def test_gradient_verification_rejects_finite_nan_mismatch() -> None:
    inputs = (torch.zeros(4),)
    graph = {"source_input_indices": [0]}

    with pytest.raises(GradientVerificationError, match="non-finite"):
        verify_gradients(
            _nan_gradient,
            cast(IRGraphExecutor, lambda value: value * 0),
            graph,
            inputs,
            _policy(),
        )

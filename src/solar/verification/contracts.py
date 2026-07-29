# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Validated numerical and case-generation policies for IR verification."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class TolerancePolicy:
    """Numerical acceptance policy for graph verification."""

    atol: float
    rtol: float
    required_matched_ratio: float = 1.0
    max_error_cap: float | None = None
    allow_negative_inf: bool = False

    def __post_init__(self) -> None:
        """Reject invalid numerical acceptance thresholds at construction."""
        values = [self.atol, self.rtol, self.required_matched_ratio]
        if self.max_error_cap is not None:
            values.append(self.max_error_cap)
        if not all(math.isfinite(value) and value >= 0 for value in values):
            raise ValueError(
                "verification tolerances must be finite and non-negative",
            )
        if self.required_matched_ratio > 1:
            raise ValueError("required_matched_ratio cannot exceed one")


@dataclass(frozen=True)
class VerificationPolicy(TolerancePolicy):
    """Case generation and execution policy for graph verification."""

    seeds: Sequence[int] = (11, 29, 47)
    patterns: Sequence[str] = ("random", "zeros", "boundary")
    device: str = "cpu"
    preserved_input_indices: Sequence[int] = ()

    def __post_init__(self) -> None:
        """Validate tolerances and structured-input protection indices."""
        super().__post_init__()
        indices = tuple(int(index) for index in self.preserved_input_indices)
        if any(index < 0 for index in indices) or len(indices) != len(
            set(indices)
        ):
            raise ValueError(
                "preserved_input_indices must be unique and non-negative",
            )
        object.__setattr__(self, "preserved_input_indices", indices)


__all__ = ["TolerancePolicy", "VerificationPolicy"]

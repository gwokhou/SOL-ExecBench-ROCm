# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Validated numerical and case-generation policies for IR verification."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from solar.ir.contracts import IRBackend
from solar.types import DynamicValue

LayerExecutor = Callable[
    [
        str,
        Mapping[str, DynamicValue],
        Sequence[DynamicValue],
        Sequence[tuple[int, ...]],
    ],
    DynamicValue,
]


@dataclass(frozen=True)
class IRVerificationBackend:
    """Verification runtime paired with its representation backend."""

    ir: IRBackend
    execute: LayerExecutor


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
    verify_gradients: bool = True
    gradient_input_indices: Sequence[int] | None = None
    gradient_atol: float | None = None
    gradient_rtol: float | None = None

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
        gradient_indices = self.gradient_input_indices
        if gradient_indices is not None:
            normalized = tuple(int(index) for index in gradient_indices)
            if any(index < 0 for index in normalized) or len(normalized) != len(
                set(normalized)
            ):
                raise ValueError(
                    "gradient_input_indices must be unique and non-negative"
                )
            object.__setattr__(self, "gradient_input_indices", normalized)
        for name, value in (
            ("gradient_atol", self.gradient_atol),
            ("gradient_rtol", self.gradient_rtol),
        ):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and non-negative")


__all__ = [
    "IRVerificationBackend",
    "LayerExecutor",
    "TolerancePolicy",
    "VerificationPolicy",
]

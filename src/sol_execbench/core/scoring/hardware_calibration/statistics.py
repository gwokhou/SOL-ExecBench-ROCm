# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Conservative, reproducible calibration statistics."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Sequence


MINIMUM_SAMPLE_COUNT = 7
RETAINED_SAMPLE_COUNT = 3
MAX_RETAINED_SPREAD = 0.05


@dataclass(frozen=True)
class ConservativeSampleSelection:
    """The best samples and their deliberately conservative representative."""

    value: float
    retained_samples: tuple[float, ...]
    spread: float


def select_conservative_value(samples: Sequence[float]) -> ConservativeSampleSelection:
    """Return the minimum of the three best stable, positive samples."""
    normalized = tuple(float(sample) for sample in samples)
    if len(normalized) < MINIMUM_SAMPLE_COUNT:
        raise ValueError(f"at least {MINIMUM_SAMPLE_COUNT} samples are required")
    if any(not isfinite(sample) or sample <= 0.0 for sample in normalized):
        raise ValueError("samples must be finite positive numbers")
    retained = tuple(sorted(normalized, reverse=True)[:RETAINED_SAMPLE_COUNT])
    spread = (retained[0] - retained[-1]) / retained[-1]
    if spread > MAX_RETAINED_SPREAD:
        raise ValueError("retained sample spread must be <= 0.05")
    return ConservativeSampleSelection(
        value=retained[-1], retained_samples=retained, spread=spread
    )

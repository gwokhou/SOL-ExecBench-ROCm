# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Confidence levels for platform capability evidence."""

from __future__ import annotations

from enum import Enum


class EstimateConfidence(str, Enum):
    """Confidence level for hardware, graph, and bound estimates."""

    SUPPORTED = "supported"
    INEXACT = "inexact"
    UNSUPPORTED = "unsupported"


__all__ = ["EstimateConfidence"]

# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Shared confidence levels for AMD scoring estimates."""

from __future__ import annotations

from enum import Enum


class EstimateConfidence(str, Enum):
    """Confidence level for hardware, graph, and bound estimates."""

    SUPPORTED = "supported"
    INEXACT = "inexact"
    UNSUPPORTED = "unsupported"

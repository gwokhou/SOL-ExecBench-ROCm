# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Shared CLI profiling-mode vocabulary."""

from __future__ import annotations

from enum import Enum


class ProfileMode(str, Enum):
    """Optional profiling modes accepted by the evaluation command."""

    NONE = "none"
    ROCPROFV3 = "rocprofv3"


PROFILE_NONE = ProfileMode.NONE.value
PROFILE_ROCPROFV3 = ProfileMode.ROCPROFV3.value

__all__ = ["PROFILE_NONE", "PROFILE_ROCPROFV3", "ProfileMode"]

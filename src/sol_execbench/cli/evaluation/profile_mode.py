# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Shared CLI profiling-mode vocabulary."""

from __future__ import annotations

from enum import StrEnum


class ProfileMode(StrEnum):
    """Optional profiling modes accepted by the evaluation command."""

    NONE = "none"
    ROCPROFV3 = "rocprofv3"


# Click choices require exact built-in strings rather than StrEnum instances.
PROFILE_NONE = str(ProfileMode.NONE)
PROFILE_ROCPROFV3 = str(ProfileMode.ROCPROFV3)

__all__ = ["PROFILE_NONE", "PROFILE_ROCPROFV3", "ProfileMode"]

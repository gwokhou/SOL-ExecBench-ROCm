# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Shared CLI sidecar-mode vocabulary."""

from enum import StrEnum


class SidecarMode(StrEnum):
    """Optional diagnostic sidecar modes accepted by evaluation."""

    NONE = "none"
    AUTO = "auto"


# Click choices require exact built-in strings rather than StrEnum instances.
SIDECAR_MODE_CHOICES = tuple(map(str, SidecarMode))

__all__ = ["SIDECAR_MODE_CHOICES", "SidecarMode"]

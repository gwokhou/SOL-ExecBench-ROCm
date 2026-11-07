# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Current artifact schemas owned by the AMD ISA integration."""

from enum import StrEnum


class AMDISAArtifactSchema(StrEnum):
    """Canonical identifiers for independently serialized AMD ISA data."""

    RELEASE_LOCK = "sol_execbench.amd_isa_release_lock.v1"


__all__ = ["AMDISAArtifactSchema"]

# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Current content identity for controlled rocprofv3 counter replay."""

from __future__ import annotations

from typing import Literal

from sol_execbench.core.bench.rocm_profiler.schema_versions import (
    ProfilerArtifactSchema,
)
from sol_execbench.core.data.base_model import CurrentFrozenSchemaModel
from sol_execbench.core.integrity import SHA256Digest


class Rocprofv3CounterProvenance(CurrentFrozenSchemaModel):
    """Hashes for every executable and configuration admitted to replay."""

    current_schema_version = ProfilerArtifactSchema.ROCPROFV3_COUNTER_PROVENANCE

    schema_version: Literal[
        ProfilerArtifactSchema.ROCPROFV3_COUNTER_PROVENANCE
    ] = ProfilerArtifactSchema.ROCPROFV3_COUNTER_PROVENANCE
    diagnostic_only: Literal[True] = True
    score_authority: Literal[False] = False
    replay_phase: Literal["evidence"] = "evidence"
    profiler_sha256: SHA256Digest | Literal["unresolved"]
    counter_definition_sha256: SHA256Digest
    configuration_sha256: SHA256Digest
    availability_sha256: SHA256Digest
    pmc_check_sha256: SHA256Digest
    application_executable_sha256: SHA256Digest | Literal["unresolved"]
    application_command_sha256: SHA256Digest


__all__ = ["Rocprofv3CounterProvenance"]

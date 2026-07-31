# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Current content identity for controlled rocprofv3 counter replay."""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict

from sol_execbench.core.data.base_model import CurrentSchemaModel
from sol_execbench.core.integrity import SHA256Digest
from sol_execbench.core.integrity.schema_versions import (
    ROCPROFV3_COUNTER_PROVENANCE_SCHEMA_VERSION,
)


class Rocprofv3CounterProvenance(CurrentSchemaModel):
    """Hashes for every executable and configuration admitted to replay."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    current_schema_version = ROCPROFV3_COUNTER_PROVENANCE_SCHEMA_VERSION

    schema_version: Literal["sol_execbench.rocprofv3_counter_provenance.v5"] = (
        ROCPROFV3_COUNTER_PROVENANCE_SCHEMA_VERSION
    )
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

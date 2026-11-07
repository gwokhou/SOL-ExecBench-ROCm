# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Audit-only aggregate of canonical domain-owned artifact schemas."""

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Final

from sol_execbench.core.bench.performance_model.diagnostic_schema_versions import (
    DiagnosticArtifactSchema,
)
from sol_execbench.core.bench.performance_model.lifecycle.schema_versions import (
    DiagnosticLifecycleSchema,
)
from sol_execbench.core.bench.performance_model.schema_versions import (
    PerformanceArtifactSchema,
)
from sol_execbench.core.bench.rocm_profiler.schema_versions import (
    ProfilerArtifactSchema,
)
from sol_execbench.core.control_plane_schema_versions import (
    ExecutionControlSchema,
)
from sol_execbench.core.data.schema_versions import BenchmarkArtifactSchema
from sol_execbench.core.dataset.schema_versions import (
    CURRENT_NUMERIC_DATASET_SCHEMAS,
    DatasetArtifactSchema,
)
from sol_execbench.core.platform.schema_versions import PlatformArtifactSchema
from sol_execbench.core.scoring.schema_versions import ReleaseArtifactSchema
from sol_execbench.tools.amd_isa.schema_versions import AMDISAArtifactSchema

ARTIFACT_SCHEMA_REGISTRIES: Final[tuple[type[StrEnum], ...]] = (
    BenchmarkArtifactSchema,
    DatasetArtifactSchema,
    PlatformArtifactSchema,
    ExecutionControlSchema,
    ProfilerArtifactSchema,
    PerformanceArtifactSchema,
    DiagnosticArtifactSchema,
    DiagnosticLifecycleSchema,
    ReleaseArtifactSchema,
    AMDISAArtifactSchema,
)

ARTIFACT_SCHEMA_MEMBERS: Final[tuple[StrEnum, ...]] = tuple(
    member for registry in ARTIFACT_SCHEMA_REGISTRIES for member in registry
)
CURRENT_STRING_ARTIFACT_SCHEMAS: Final[frozenset[str]] = frozenset(
    member.value for member in ARTIFACT_SCHEMA_MEMBERS
)
CURRENT_NUMERIC_ARTIFACT_SCHEMAS: Final[Mapping[str, int]] = MappingProxyType(
    dict(CURRENT_NUMERIC_DATASET_SCHEMAS)
)

__all__ = [
    "ARTIFACT_SCHEMA_MEMBERS",
    "ARTIFACT_SCHEMA_REGISTRIES",
    "CURRENT_NUMERIC_ARTIFACT_SCHEMAS",
    "CURRENT_STRING_ARTIFACT_SCHEMAS",
]

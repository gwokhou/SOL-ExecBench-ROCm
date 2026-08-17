# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Current dataset and corpus artifact schemas."""

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Final


class DatasetArtifactSchema(StrEnum):
    """Canonical dataset governance and corpus identifiers."""

    CORPUS_MANIFEST = "sol_execbench.corpus_manifest.v3"
    CORPUS_TARGET_VIEW = "sol_execbench.corpus_target_view.v3"
    WORKLOAD_GENERATION_RULE = "sol_execbench.workload_generation_rule.v2"
    CORPUS_AGENT_VIEW = "sol_execbench.corpus_agent_view.v1"
    HARDWARE_GENERALIZATION = "sol_execbench.hardware_generalization.v1"
    CORPUS_READINESS = "sol_execbench.corpus_readiness.v1"
    DATASET_GOVERNANCE = "sol_execbench.dataset_governance.v1"


class CorpusReadinessArtifactKind(StrEnum):
    """Artifacts emitted by one corpus readiness audit."""

    RECORD = "record"
    SUMMARY = "summary"


class DatasetGovernanceArtifactKind(StrEnum):
    """Dataset policy and policy-check result variants."""

    POLICY = "policy"
    REDISTRIBUTION_CHECK = "redistribution_check"


AKA_CORPUS_MANIFEST_SCHEMA_VERSION: Final = 7
AKA_MATERIALIZATION_MANIFEST_SCHEMA_VERSION: Final = 2
AKA_TOLERANCE_CALIBRATION_SCHEMA_VERSION: Final = 3
COVERAGE_POLICY_SCHEMA_VERSION: Final = 1

CURRENT_NUMERIC_DATASET_SCHEMAS: Final[Mapping[str, int]] = MappingProxyType(
    {
        "aka_corpus_manifest": AKA_CORPUS_MANIFEST_SCHEMA_VERSION,
        "aka_materialization_manifest": (
            AKA_MATERIALIZATION_MANIFEST_SCHEMA_VERSION
        ),
        "aka_tolerance_calibration": AKA_TOLERANCE_CALIBRATION_SCHEMA_VERSION,
        "coverage_policy": COVERAGE_POLICY_SCHEMA_VERSION,
    }
)

__all__ = [
    "AKA_CORPUS_MANIFEST_SCHEMA_VERSION",
    "AKA_MATERIALIZATION_MANIFEST_SCHEMA_VERSION",
    "AKA_TOLERANCE_CALIBRATION_SCHEMA_VERSION",
    "COVERAGE_POLICY_SCHEMA_VERSION",
    "CURRENT_NUMERIC_DATASET_SCHEMAS",
    "CorpusReadinessArtifactKind",
    "DatasetArtifactSchema",
    "DatasetGovernanceArtifactKind",
]

# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Content-addressed development and held-out diagnostic corpora."""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from sol_execbench.core.bench.performance_model.models import WorkloadKind
from sol_execbench.core.data.base_model import (
    CurrentSchemaModel,
    StrictArtifactModel,
)
from sol_execbench.core.integrity import SHA256Digest
from sol_execbench.core.integrity.schema_versions import (
    DIAGNOSTIC_VALIDATION_CORPUS_SCHEMA_VERSION,
)

MINIMUM_CASES_PER_FAMILY = 20
MINIMUM_CORPUS_CASES = 80
_SUPPORTED_FAMILIES = frozenset(
    {
        WorkloadKind.ELEMENTWISE,
        WorkloadKind.TRANSPOSE,
        WorkloadKind.REDUCTION,
        WorkloadKind.MATMUL,
    }
)
_CONFIG = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class ValidationArtifactReference(StrictArtifactModel):
    """One immutable artifact input to a validation case."""

    model_config = _CONFIG

    path: str = Field(min_length=1)
    sha256: SHA256Digest


class DiagnosticValidationCase(StrictArtifactModel):
    """Labels and source references, deliberately excluding model output."""

    model_config = _CONFIG

    case_id: str = Field(min_length=1)
    pair_id: str = Field(min_length=1)
    workload_kind: WorkloadKind
    evidence_manifest: ValidationArtifactReference
    solar_manifest: ValidationArtifactReference
    gold_action_codes: list[str] = Field(default_factory=list)
    independent: Literal[True] = True


class DiagnosticValidationCorpus(CurrentSchemaModel):
    """Frozen development or held-out cases for the four admitted families."""

    model_config = _CONFIG
    current_schema_version = DIAGNOSTIC_VALIDATION_CORPUS_SCHEMA_VERSION

    schema_version: Literal["sol_execbench.diagnostic_validation_corpus.v1"] = (
        DIAGNOSTIC_VALIDATION_CORPUS_SCHEMA_VERSION
    )
    role: Literal["development", "held_out"]
    cases: list[DiagnosticValidationCase] = Field(
        min_length=MINIMUM_CORPUS_CASES
    )

    @model_validator(mode="after")
    def cases_are_complete_and_unique(self) -> DiagnosticValidationCorpus:
        """Require unique cases and at least twenty from each family."""
        case_ids = [case.case_id for case in self.cases]
        pair_ids = [case.pair_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("validation corpus repeats case_id")
        if len(pair_ids) != len(set(pair_ids)):
            raise ValueError(
                "validation corpus repeats workload/candidate pair"
            )
        kinds = {case.workload_kind for case in self.cases}
        if not kinds <= _SUPPORTED_FAMILIES:
            raise ValueError("validation corpus contains unsupported family")
        for kind in _SUPPORTED_FAMILIES:
            count = sum(case.workload_kind is kind for case in self.cases)
            if count < MINIMUM_CASES_PER_FAMILY:
                raise ValueError(f"validation corpus lacks {kind} coverage")
        return self


def require_disjoint_corpora(
    development: DiagnosticValidationCorpus,
    held_out: DiagnosticValidationCorpus,
) -> None:
    """Reject role mistakes or repeated workload/candidate pairs."""
    if development.role != "development" or held_out.role != "held_out":
        raise ValueError("validation corpus roles are invalid")
    overlap = {case.pair_id for case in development.cases} & {
        case.pair_id for case in held_out.cases
    }
    if overlap:
        raise ValueError("development and held-out corpora overlap")


__all__ = [
    "MINIMUM_CASES_PER_FAMILY",
    "MINIMUM_CORPUS_CASES",
    "DiagnosticValidationCase",
    "DiagnosticValidationCorpus",
    "ValidationArtifactReference",
    "require_disjoint_corpora",
]

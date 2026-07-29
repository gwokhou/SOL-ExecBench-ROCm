# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Governed held-out acceptance for the gfx1200 diagnostic model."""

from __future__ import annotations

import statistics
from typing import Literal

import numpy as np
from pydantic import ConfigDict, Field, model_validator

from sol_execbench.core.bench.performance_model.models import (
    PERFORMANCE_MODEL_VERSION,
    CalibrationIdentity,
    WorkloadKind,
)
from sol_execbench.core.data.base_model import BaseModelWithDocstrings
from sol_execbench.core.integrity import SHA256Digest, stable_json_checksum
from sol_execbench.core.integrity.schema_versions import (
    DIAGNOSTIC_ACCEPTANCE_SCHEMA_VERSION,
)

_MODEL_CONFIG = ConfigDict(
    extra="forbid",
    frozen=True,
    allow_inf_nan=False,
)
_EXPECTED_PRIMARY_CODES = {
    WorkloadKind.ELEMENTWISE: {"launch_bound"},
    WorkloadKind.TRANSPOSE: {"memory_access_efficiency"},
    WorkloadKind.REDUCTION: {"lds_barrier_pressure"},
    WorkloadKind.MATMUL: {"matrix_path_missing"},
}
MINIMUM_CASES_PER_FAMILY = 10


class DiagnosticAcceptanceCase(BaseModelWithDocstrings):
    """One independent workload result excluded from calibration."""

    model_config = _MODEL_CONFIG

    workload_uuid: str = Field(min_length=1)
    workload_kind: WorkloadKind
    candidate_sha256: SHA256Digest
    evidence_manifest_sha256: SHA256Digest
    performance_diagnostic_sha256: SHA256Digest
    independent_held_out: Literal[True] = True
    tuning_sample: Literal[False] = False
    predicted_ms: float = Field(gt=0.0)
    measured_ms: float = Field(gt=0.0)
    primary_attribution_code: str = Field(min_length=1)


class DiagnosticAcceptanceManifest(BaseModelWithDocstrings):
    """Frozen identity and held-out cases supplied to acceptance."""

    model_config = _MODEL_CONFIG

    schema_version: Literal["sol_execbench.diagnostic_acceptance.v1"] = (
        DIAGNOSTIC_ACCEPTANCE_SCHEMA_VERSION
    )
    model_version: Literal["gfx1200_diagnostic.v2"] = PERFORMANCE_MODEL_VERSION
    calibration_profile_sha256: SHA256Digest
    calibration_identity: CalibrationIdentity
    frozen_configuration_sha256: SHA256Digest
    tuning_evidence_sha256: list[SHA256Digest] = Field(min_length=1)
    configuration_frozen_before_acceptance: Literal[True] = True
    cases: list[DiagnosticAcceptanceCase] = Field(min_length=40)

    @model_validator(mode="after")
    def cases_are_independent(self) -> DiagnosticAcceptanceManifest:
        """Reject duplicate workload/candidate pairs."""
        identities = [
            (case.workload_uuid, case.candidate_sha256) for case in self.cases
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("acceptance cases repeat workload/candidate")
        return self


class DiagnosticAcceptanceResult(BaseModelWithDocstrings):
    """Content-bound aggregate acceptance verdict."""

    model_config = _MODEL_CONFIG

    schema_version: Literal["sol_execbench.diagnostic_acceptance.v1"] = (
        DIAGNOSTIC_ACCEPTANCE_SCHEMA_VERSION
    )
    model_version: Literal["gfx1200_diagnostic.v2"] = PERFORMANCE_MODEL_VERSION
    manifest_sha256: SHA256Digest
    calibration_profile_sha256: SHA256Digest
    calibration_identity: CalibrationIdentity
    accepted: bool
    case_count: int = Field(ge=0)
    family_case_counts: dict[WorkloadKind, int]
    median_absolute_percentage_error: float = Field(ge=0.0)
    p90_absolute_percentage_error: float = Field(ge=0.0)
    attribution_matches: dict[WorkloadKind, bool]
    reason_codes: list[str] = Field(default_factory=list)


def evaluate_diagnostic_acceptance(
    manifest: DiagnosticAcceptanceManifest,
) -> DiagnosticAcceptanceResult:
    """Apply frozen accuracy, coverage, and attribution thresholds."""
    cases = manifest.cases
    counts = {
        kind: sum(case.workload_kind is kind for case in cases)
        for kind in _EXPECTED_PRIMARY_CODES
    }
    reasons = [
        "acceptance_workload_coverage_incomplete"
        for kind in counts
        if counts[kind] < MINIMUM_CASES_PER_FAMILY
    ]
    errors = [
        abs(case.predicted_ms - case.measured_ms) / case.measured_ms * 100.0
        for case in cases
    ]
    median_error = statistics.median(errors) if errors else 0.0
    p90_error = float(np.percentile(errors, 90.0)) if errors else 0.0
    if median_error > 15.0:
        reasons.append("median_absolute_percentage_error_exceeded")
    if p90_error > 30.0:
        reasons.append("p90_absolute_percentage_error_exceeded")
    attribution_matches = _attribution_matches(cases)
    if not all(attribution_matches.values()):
        reasons.append("primary_attribution_mismatch")
    return DiagnosticAcceptanceResult(
        manifest_sha256=stable_json_checksum(manifest.model_dump(mode="json")),
        calibration_profile_sha256=manifest.calibration_profile_sha256,
        calibration_identity=manifest.calibration_identity,
        accepted=not reasons,
        case_count=len(cases),
        family_case_counts=counts,
        median_absolute_percentage_error=median_error,
        p90_absolute_percentage_error=p90_error,
        attribution_matches=attribution_matches,
        reason_codes=list(dict.fromkeys(reasons)),
    )


def _attribution_matches(
    cases: list[DiagnosticAcceptanceCase],
) -> dict[WorkloadKind, bool]:
    result: dict[WorkloadKind, bool] = {}
    for kind, expected_codes in _EXPECTED_PRIMARY_CODES.items():
        kind_cases = [case for case in cases if case.workload_kind is kind]
        result[kind] = len(kind_cases) >= MINIMUM_CASES_PER_FAMILY and all(
            case.primary_attribution_code in expected_codes
            for case in kind_cases
        )
    return result


__all__ = [
    "MINIMUM_CASES_PER_FAMILY",
    "DiagnosticAcceptanceCase",
    "DiagnosticAcceptanceManifest",
    "DiagnosticAcceptanceResult",
    "evaluate_diagnostic_acceptance",
]

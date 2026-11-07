# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Validate callable verification receipts without importing their reference."""

from __future__ import annotations

import re
import sys
from collections.abc import Mapping
from typing import TypeGuard

from solar.artifacts import ArtifactDocument, ArtifactMap, stable_json_checksum
from solar.schema_versions import SchemaVersion
from solar.verification.contracts import (
    IN_TOTO_STATEMENT_TYPE,
    VerificationPolicy,
    VerificationPredicate,
)
from solar.verification.errors import VerificationError
from solar.verification.execution import (
    ExecutionBackend,
    require_current_semantic_runtime,
)
from solar.verification.policy import verification_policy_record


def _subjects(document: ArtifactDocument) -> dict[str, str]:
    values = document.get("subject")
    if not isinstance(values, list):
        raise VerificationError("verification subjects must be a list")
    result: dict[str, str] = {}
    for value in values:
        if not isinstance(value, dict):
            raise VerificationError("verification subject is malformed")
        raw_digest = value.get("digest")
        if not isinstance(raw_digest, dict):
            raise VerificationError("verification subject digest is malformed")
        name, digest = value.get("name"), raw_digest.get("sha256")
        if (
            not isinstance(name, str)
            or not isinstance(digest, str)
            or name in result
        ):
            raise VerificationError(
                "verification subjects lack unique names and digests"
            )
        result[name] = digest
    return result


def _require_results(
    predicate: ArtifactMap, policy: VerificationPolicy, cases: list[ArtifactMap]
) -> None:
    if predicate.get("cases") != cases:
        raise VerificationError("verification case policy differs")
    results = predicate.get("results")
    if not isinstance(results, list) or len(results) != len(cases):
        raise VerificationError("verification results are incomplete")
    for case, result in zip(cases, results, strict=True):
        if not isinstance(result, dict) or any(
            result.get(key) != value
            for key, value in {
                "seed": case["seed"],
                "pattern": case["pattern"],
                "parameters_sha256": stable_json_checksum({}),
            }.items()
        ):
            raise VerificationError(
                "verification result is not bound to its case"
            )
        error = result.get("max_abs_error")
        if not _finite_nonnegative(error):
            raise VerificationError(
                "verification result lacks finite error evidence"
            )
        matched = result.get("matched_ratio")
        if matched is not None and (
            not _finite_nonnegative(matched) or matched > 1
        ):
            raise VerificationError(
                "verification result has invalid matched ratio"
            )
        gradient_count = result.get("gradient_inputs_verified")
        gradient_error = result.get("gradient_error_semantics_verified")
        if policy.verify_gradients and not (
            _finite_nonnegative(gradient_count)
            or (_finite_nonnegative(gradient_error) and gradient_error == 1)
        ):
            raise VerificationError(
                "verification result lacks gradient evidence"
            )


def _finite_nonnegative(value: object) -> TypeGuard[int | float]:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and 0 <= value <= sys.float_info.max
    )


def _require_execution_shape(
    execution: ArtifactMap, policy: VerificationPolicy
) -> None:
    try:
        backend = ExecutionBackend(execution.get("backend"))
    except ValueError as exc:
        raise VerificationError(
            "verification execution backend is invalid"
        ) from exc
    device_type = "cpu" if backend is ExecutionBackend.CPU else "cuda"
    device = execution.get("device")
    if (
        execution.get("device_type") != device_type
        or policy.device.split(":", 1)[0] != device_type
        or not isinstance(device, str)
        or re.fullmatch(rf"{device_type}(?::[0-9]+)?", device) is None
    ):
        raise VerificationError(
            "verification execution device identity is invalid"
        )
    if backend is not ExecutionBackend.CPU:
        name = execution.get("device_name")
        if not isinstance(name, str) or not name:
            raise VerificationError(
                "verification execution device name is missing"
            )
    if backend is ExecutionBackend.ROCM:
        hip, gfx = execution.get("hip_version"), execution.get("gfx_target")
        if (
            not isinstance(hip, str)
            or not hip
            or not isinstance(gfx, str)
            or not gfx.startswith("gfx")
        ):
            raise VerificationError(
                "verification ROCm execution identity is incomplete"
            )


def validate_callable_receipt(
    document: ArtifactDocument,
    *,
    subjects: Mapping[str, str],
    policy: VerificationPolicy,
    cases: list[ArtifactMap],
    execution_sha256: str,
) -> ArtifactMap:
    """Require current proof identity, exact policy, runtime and complete cases."""
    if (
        document.get("_type") != IN_TOTO_STATEMENT_TYPE
        or document.get("predicateType") != VerificationPredicate.CALLABLE
    ):
        raise VerificationError("unsupported callable verification statement")
    predicate = document.require_mapping("predicate")
    if predicate.get("verifier") != SchemaVersion.IR_VERIFICATION:
        raise VerificationError("verification receipt schema is not current")
    if predicate.get("status") != "passed" or _subjects(document) != subjects:
        raise VerificationError(
            "verification receipt status or subjects differ"
        )
    if predicate.get("tolerance") != verification_policy_record(policy):
        raise VerificationError(
            "verification numerical or gradient policy differs"
        )
    execution = predicate.get("execution")
    if (
        not isinstance(execution, dict)
        or stable_json_checksum(execution) != execution_sha256
    ):
        raise VerificationError("verification execution identity differs")
    _require_execution_shape(execution, policy)
    require_current_semantic_runtime(execution)
    _require_results(predicate, policy, cases)
    return execution

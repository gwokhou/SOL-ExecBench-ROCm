# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0
"""Core schema/reference readiness classification handlers."""

from __future__ import annotations

from .readiness_classification_state import ReadinessClassificationState
from .readiness_hints import _unsupported_dtype_failure
from .readiness_models import ReadinessClass

def classify_unsupported_dtype(state: ReadinessClassificationState) -> bool:
    unsupported_dtype = _unsupported_dtype_failure(state.problem, state.workload)
    if unsupported_dtype is None:
        return False
    state.layers.schema_known = "blocked"
    state.status = "dtype_blocked"
    state.readiness_class = ReadinessClass.UNSUPPORTED
    message = f"Unsupported dtype in migrated schema: {unsupported_dtype}"
    next_action = "Exclude or add an explicit ROCm dtype compatibility path."
    state.add_reason(
        "unsupported_dtype",
        message,
        next_action,
        state.problem.problem_path,
    )
    state.add_blocker(
        code="unsupported_dtype",
        blocker_type="unsupported_dtype",
        message=message,
        next_action=next_action,
        evidence_path=state.problem.problem_path,
    )
    return True


def classify_schema_failure(state: ReadinessClassificationState) -> bool:
    if state.problem.schema_status == "parsed" and state.workload.schema_status == "parsed":
        return False
    state.layers.schema_known = "blocked"
    state.status = "schema_input_blocked"
    state.readiness_class = ReadinessClass.BLOCKED_MISSING_EVIDENCE
    code = (
        "workload_schema_failure"
        if state.workload.schema_status != "parsed"
        else "definition_schema_failure"
    )
    message = (
        state.workload.schema_failure
        or state.problem.schema_failure
        or "schema parse failed"
    )
    next_action = "Fix or exclude malformed canonical dataset entry."
    state.add_reason(code, message, next_action, state.problem.problem_path)
    state.add_blocker(
        code=code,
        blocker_type="missing_evidence",
        message=message,
        next_action=next_action,
        evidence_path=state.problem.problem_path,
    )
    return True


def classify_missing_reference(state: ReadinessClassificationState) -> bool:
    if state.problem.reference_available:
        return False
    state.status = "schema_input_blocked"
    state.readiness_class = ReadinessClass.BLOCKED_MISSING_EVIDENCE
    state.layers.reference_execution = "blocked"
    message = "Reference source is not available."
    next_action = "Restore reference source before execution attempts."
    state.add_reason(
        "missing_reference",
        message,
        next_action,
        state.problem.problem_path,
    )
    state.add_blocker(
        code="missing_reference",
        blocker_type="missing_evidence",
        message=message,
        next_action=next_action,
        evidence_path=state.problem.problem_path,
    )
    return True

def classify_custom_inputs(state: ReadinessClassificationState) -> bool:
    if not state.workload.uses_custom_inputs:
        return False
    if state.problem.definition and state.problem.definition.custom_inputs_entrypoint:
        state.layers.input_generation = "ready_to_generate"
        return True
    state.status = "custom_input_blocked"
    state.readiness_class = ReadinessClass.BLOCKED_MISSING_EVIDENCE
    state.layers.input_generation = "blocked"
    message = "Custom input workload is missing a valid definition.custom_inputs_entrypoint."
    next_action = (
        "Restore the benchmark-defined custom input entrypoint before execution attempts."
    )
    state.add_reason(
        "custom_input_requires_evaluator_support",
        message,
        next_action,
        state.problem.problem_path,
    )
    state.add_blocker(
        code="custom_input_requires_evaluator_support",
        blocker_type="missing_evidence",
        message=message,
        next_action=next_action,
        evidence_path=state.problem.problem_path,
    )
    return True

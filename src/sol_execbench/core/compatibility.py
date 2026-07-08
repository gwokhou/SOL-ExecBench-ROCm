# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Strict diagnostic-only ROCm compatibility Matrix Entry contract facade."""

from __future__ import annotations

from sol_execbench.core.compatibility_entry_models import (
    MatrixClaimBoundary,
    MatrixEntry,
    MatrixExecutionDecision,
    RocmCompatibilityMatrixReport,
)
from sol_execbench.core.compatibility_enums import (
    MATRIX_ENTRY_JSON_SCHEMA_ID,
    ROCM_COMPATIBILITY_MATRIX_REPORT_JSON_SCHEMA_ID,
    ROCM_COMPATIBILITY_MATRIX_SCHEMA_VERSION,
    MatrixCompatibilityReasonCode,
    MatrixCompatibilityReasonCodeField,
    MatrixCompatibilityStatus,
    MatrixCompatibilityStatusField,
    MatrixValidationScope,
    MatrixValidationScopeField,
)
from sol_execbench.core.compatibility_evidence_models import (
    MatrixArtifactReference,
    MatrixContainerEvidence,
    MatrixDependencyPolicyEvidence,
    MatrixGpuEvidence,
    MatrixHostEvidence,
    MatrixObservedEvidence,
    MatrixPythonDependencyEvidence,
    MatrixTarget,
    MatrixToolchainEvidence,
)
from sol_execbench.core.compatibility_execution import (
    build_matrix_entry,
    classify_matrix_entry_for_execution,
)
from sol_execbench.core.compatibility_schema import (
    export_matrix_entry_json_schema,
    export_matrix_json_schemas,
    export_rocm_compatibility_matrix_report_json_schema,
)

__all__ = [
    "MATRIX_ENTRY_JSON_SCHEMA_ID",
    "ROCM_COMPATIBILITY_MATRIX_REPORT_JSON_SCHEMA_ID",
    "ROCM_COMPATIBILITY_MATRIX_SCHEMA_VERSION",
    "MatrixArtifactReference",
    "MatrixClaimBoundary",
    "MatrixCompatibilityReasonCode",
    "MatrixCompatibilityReasonCodeField",
    "MatrixCompatibilityStatus",
    "MatrixCompatibilityStatusField",
    "MatrixContainerEvidence",
    "MatrixDependencyPolicyEvidence",
    "MatrixEntry",
    "MatrixExecutionDecision",
    "MatrixGpuEvidence",
    "MatrixHostEvidence",
    "MatrixObservedEvidence",
    "MatrixPythonDependencyEvidence",
    "MatrixTarget",
    "MatrixToolchainEvidence",
    "MatrixValidationScope",
    "MatrixValidationScopeField",
    "RocmCompatibilityMatrixReport",
    "build_matrix_entry",
    "classify_matrix_entry_for_execution",
    "export_matrix_entry_json_schema",
    "export_matrix_json_schemas",
    "export_rocm_compatibility_matrix_report_json_schema",
]

# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Compatibility matrix constants, enums, and strict field aliases."""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BeforeValidator, ConfigDict

from sol_execbench.core.integrity.schema_versions import (
    ROCM_COMPATIBILITY_MATRIX_SCHEMA_VERSION,
)


MATRIX_ENTRY_JSON_SCHEMA_ID = (
    "https://sol-execbench.local/schemas/"
    f"{ROCM_COMPATIBILITY_MATRIX_SCHEMA_VERSION}.matrix_entry.schema.json"
)
ROCM_COMPATIBILITY_MATRIX_REPORT_JSON_SCHEMA_ID = (
    "https://sol-execbench.local/schemas/"
    f"{ROCM_COMPATIBILITY_MATRIX_SCHEMA_VERSION}.report.schema.json"
)
MATRIX_MODEL_CONFIG = ConfigDict(
    extra="forbid",
    frozen=True,
    strict=True,
    use_attribute_docstrings=True,
)


class MatrixCompatibilityStatus(str, Enum):
    """Bounded compatibility status vocabulary for Matrix Entries."""

    HOST_VALIDATED = "host_validated"
    CONTAINER_VALIDATED = "container_validated"
    MIXED_VERSION = "mixed_version"
    PYTORCH_WHEEL_UNAVAILABLE = "pytorch_wheel_unavailable"
    RUNTIME_UNAVAILABLE = "runtime_unavailable"
    NOT_TESTED = "not_tested"


class MatrixCompatibilityReasonCode(str, Enum):
    """Stable reason-code vocabulary for compatibility Matrix Entries."""

    HOST_NATIVE_VALIDATED = "host_native_validated"
    CONTAINER_USER_SPACE_VALIDATED = "container_user_space_validated"
    TARGET_OBSERVED_MISMATCH = "target_observed_mismatch"
    PYTORCH_ROCM_WHEEL_UNAVAILABLE = "pytorch_rocm_wheel_unavailable"
    ROCM_RUNTIME_UNAVAILABLE = "rocm_runtime_unavailable"
    TARGET_NOT_TESTED = "target_not_tested"


class MatrixValidationScope(str, Enum):
    """Requested validation scope for a Matrix Target."""

    NATIVE_HOST = "native_host"
    CONTAINER_USER_SPACE = "container_user_space"


def _validate_status(value: object) -> object:
    if isinstance(value, str):
        return MatrixCompatibilityStatus(value)
    return value


def _validate_reason_code(value: object) -> object:
    if isinstance(value, str):
        return MatrixCompatibilityReasonCode(value)
    return value


def _validate_validation_scope(value: object) -> object:
    if isinstance(value, str):
        return MatrixValidationScope(value)
    return value


MatrixCompatibilityStatusField = Annotated[
    MatrixCompatibilityStatus,
    BeforeValidator(_validate_status),
]
MatrixCompatibilityReasonCodeField = Annotated[
    MatrixCompatibilityReasonCode,
    BeforeValidator(_validate_reason_code),
]
MatrixValidationScopeField = Annotated[
    MatrixValidationScope,
    BeforeValidator(_validate_validation_scope),
]

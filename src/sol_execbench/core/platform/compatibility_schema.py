# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Compatibility matrix JSON Schema exports."""

from __future__ import annotations

from sol_execbench.core.platform.compatibility_entry_models import (
    MatrixEntry,
    RocmCompatibilityMatrixReport,
)
from sol_execbench.core.platform.compatibility_enums import (
    MATRIX_ENTRY_JSON_SCHEMA_ID,
    ROCM_COMPATIBILITY_MATRIX_REPORT_JSON_SCHEMA_ID,
    ROCM_COMPATIBILITY_MATRIX_SCHEMA_VERSION,
)


def matrix_json_schema_with_metadata(
    schema: dict[str, object],
    *,
    schema_id: str,
) -> dict[str, object]:
    enriched = dict(schema)
    enriched["$id"] = schema_id
    enriched["schema_version"] = ROCM_COMPATIBILITY_MATRIX_SCHEMA_VERSION
    enriched["x-sol-execbench-schema-version"] = (
        ROCM_COMPATIBILITY_MATRIX_SCHEMA_VERSION
    )
    return enriched


def export_matrix_entry_json_schema() -> dict[str, object]:
    """Return the strict diagnostic Matrix Entry JSON Schema."""

    return matrix_json_schema_with_metadata(
        MatrixEntry.model_json_schema(),
        schema_id=MATRIX_ENTRY_JSON_SCHEMA_ID,
    )


def export_rocm_compatibility_matrix_report_json_schema() -> dict[str, object]:
    """Return the strict ROCm Compatibility Matrix report JSON Schema."""

    return matrix_json_schema_with_metadata(
        RocmCompatibilityMatrixReport.model_json_schema(),
        schema_id=ROCM_COMPATIBILITY_MATRIX_REPORT_JSON_SCHEMA_ID,
    )


def export_matrix_json_schemas() -> dict[str, dict[str, object]]:
    """Return the Matrix schemas exported for downstream diagnostic tooling."""

    return {
        "matrix_entry": export_matrix_entry_json_schema(),
        "rocm_compatibility_matrix_report": (
            export_rocm_compatibility_matrix_report_json_schema()
        ),
    }

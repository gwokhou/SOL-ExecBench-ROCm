import json

from sol_execbench.core.compatibility import (
    ROCM_COMPATIBILITY_MATRIX_SCHEMA_VERSION,
    export_matrix_entry_json_schema,
    export_matrix_json_schemas,
    export_rocm_compatibility_matrix_report_json_schema,
)


def test_matrix_entry_schema_export_has_strict_identity_metadata():
    schema = export_matrix_entry_json_schema()

    assert schema["$id"] == (
        "https://sol-execbench.local/schemas/"
        "sol_execbench.rocm_compatibility_matrix.v1.matrix_entry.schema.json"
    )
    assert schema["title"] == "MatrixEntry"
    assert schema["schema_version"] == ROCM_COMPATIBILITY_MATRIX_SCHEMA_VERSION
    assert (
        schema["x-sol-execbench-schema-version"]
        == ROCM_COMPATIBILITY_MATRIX_SCHEMA_VERSION
    )
    assert schema["additionalProperties"] is False


def test_matrix_report_schema_export_has_strict_identity_metadata():
    schema = export_rocm_compatibility_matrix_report_json_schema()

    assert schema["$id"] == (
        "https://sol-execbench.local/schemas/"
        "sol_execbench.rocm_compatibility_matrix.v1.report.schema.json"
    )
    assert schema["title"] == "RocmCompatibilityMatrixReport"
    assert schema["schema_version"] == ROCM_COMPATIBILITY_MATRIX_SCHEMA_VERSION
    assert (
        schema["x-sol-execbench-schema-version"]
        == ROCM_COMPATIBILITY_MATRIX_SCHEMA_VERSION
    )
    assert schema["additionalProperties"] is False


def test_matrix_schema_export_scope_is_exactly_two_contracts():
    schemas = export_matrix_json_schemas()

    assert set(schemas) == {
        "matrix_entry",
        "rocm_compatibility_matrix_report",
    }


def test_matrix_schema_export_is_deterministic_with_sorted_json():
    first = json.dumps(export_matrix_json_schemas(), sort_keys=True, indent=2)
    second = json.dumps(export_matrix_json_schemas(), sort_keys=True, indent=2)

    assert first.encode("utf-8") == second.encode("utf-8")

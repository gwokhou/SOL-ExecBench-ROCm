from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from click.testing import CliRunner

from sol_execbench.cli.main import cli
from sol_execbench.core.platform.compatibility import (
    ROCM_COMPATIBILITY_MATRIX_SCHEMA_VERSION,
    export_matrix_entry_json_schema,
    export_matrix_json_schemas,
    export_rocm_compatibility_matrix_report_json_schema,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_PATH = REPO_ROOT / "scripts" / "export_matrix_schema.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("export_matrix_schema", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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


def test_export_matrix_schema_script_writes_single_model_files(tmp_path, monkeypatch):
    module = _load_script()
    entry_path = tmp_path / "entry.schema.json"
    report_path = tmp_path / "report.schema.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_matrix_schema.py",
            "--model",
            "matrix-entry",
            "--output",
            str(entry_path),
        ],
    )
    assert module.main() == 0
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_matrix_schema.py",
            "--model",
            "report",
            "--output",
            str(report_path),
        ],
    )
    assert module.main() == 0

    entry_text = entry_path.read_text(encoding="utf-8")
    report_text = report_path.read_text(encoding="utf-8")
    assert entry_text.endswith("\n")
    assert report_text.endswith("\n")
    assert json.loads(entry_text)["title"] == "MatrixEntry"
    assert json.loads(report_text)["title"] == "RocmCompatibilityMatrixReport"


def test_export_matrix_schema_script_writes_all_schema_files(tmp_path, monkeypatch):
    module = _load_script()
    output_dir = tmp_path / "schemas"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_matrix_schema.py",
            "--model",
            "all",
            "--output-dir",
            str(output_dir),
        ],
    )
    assert module.main() == 0

    assert sorted(path.name for path in output_dir.iterdir()) == [
        "matrix-entry.schema.json",
        "rocm-compatibility-matrix-report.schema.json",
    ]
    first = (output_dir / "matrix-entry.schema.json").read_text(encoding="utf-8")
    module.main()
    assert (output_dir / "matrix-entry.schema.json").read_text(
        encoding="utf-8"
    ) == first


def test_schema_export_script_is_not_primary_sol_execbench_cli_option():
    module = _load_script()
    script_parser = module.build_parser()
    assert "--model" in script_parser.format_help()

    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "export_matrix_schema" not in result.output
    assert "--model matrix-entry" not in result.output

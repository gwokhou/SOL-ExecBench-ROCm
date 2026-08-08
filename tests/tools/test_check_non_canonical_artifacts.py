# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Tests for the git-ignored curated-artifact schema gate."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from sol_execbench.core.integrity.schema_versions import SchemaVersion
from solar.schema_versions import SchemaVersion as SolarSchemaVersion

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts/check_non_canonical_artifacts.py"
)
SPEC = spec_from_file_location(
    "check_non_canonical_artifacts",
    SCRIPT_PATH,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
audit_text = MODULE.audit_text
collect_findings = MODULE.collect_findings


def test_accepts_current_schema_identifier():
    findings = audit_text(
        Path("example.json"),
        f'{{"schema_version": "{SchemaVersion.ENVIRONMENT_SNAPSHOT.value}"}}',
    )

    assert findings == []


def test_rejects_unregistered_schema_identifier():
    retired = "sol_execbench.hardware_calibration." + "v1"

    findings = audit_text(Path("legacy.json"), retired)

    assert findings == [
        f"legacy.json: unsupported schema identifier {retired}",
    ]


def test_rejects_numeric_schema_artifact():
    findings = audit_text(
        Path("artifact.json"),
        '{"schema_version": 6}',
    )

    assert findings == [
        (
            "artifact.json: unregistered numeric schema artifact "
            "(schema_version 6)"
        ),
    ]


def test_accepts_solar_current_schema_identifier():
    findings = audit_text(
        Path("audit.json"),
        (
            '{"schema_version": '
            f'"{SolarSchemaVersion.RESOURCE_PEAK_CALIBRATION.value}"}}'
        ),
    )

    assert findings == []


def test_marked_directory_is_exempt(tmp_path: Path):
    marked = tmp_path / "marked"
    marked.mkdir()
    retired_v1 = "sol_execbench.hardware_calibration." + "v1"
    retired_v2 = "sol_execbench.hardware_calibration." + "v2"
    (marked / "NON_CANONICAL.md").write_text(
        f"# Non-canonical\n`{retired_v1}`\n",
    )
    (marked / "legacy.json").write_text(
        f'{{"schema_version": "{retired_v2}"}}',
    )

    assert collect_findings(tmp_path) == []


def test_unmarked_directory_fails(tmp_path: Path):
    plain = tmp_path / "plain"
    plain.mkdir()
    retired = "sol_execbench.representative_suite." + "v1"
    (plain / "legacy.json").write_text(
        f'{{"schema_version": "{retired}"}}',
    )

    findings = collect_findings(tmp_path)

    assert findings == [
        f"plain/legacy.json: unsupported schema identifier {retired}",
    ]


def test_exempt_generated_root(tmp_path: Path):
    retired = "sol_execbench.hardware_calibration." + "v1"
    for name in (
        "cold-archive",
        "conformance",
        "outputs",
        "publications",
        "releases",
        "store",
    ):
        generated = tmp_path / name
        generated.mkdir()
        (generated / "generated.json").write_text(
            f'{{"schema_version": "{retired}"}}',
        )

    assert collect_findings(tmp_path) == []


def test_top_level_file_is_in_scope(tmp_path: Path):
    retired = "sol_execbench.fusion_validation." + "v1"
    (tmp_path / "stray.json").write_text(
        f'{{"schema_version": "{retired}"}}',
    )

    findings = collect_findings(tmp_path)

    assert len(findings) == 1
    assert "stray.json" in findings[0]

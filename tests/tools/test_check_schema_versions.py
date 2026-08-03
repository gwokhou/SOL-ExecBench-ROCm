from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from sol_execbench.core.integrity.schema_versions import SchemaVersion
from solar.schema_versions import SchemaVersion as SolarSchemaVersion

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts/check_schema_versions.py"
)
SPEC = spec_from_file_location("check_schema_versions", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
audit_text = MODULE.audit_text


def test_accepts_current_schema_identifier():
    findings, families = audit_text(
        Path("example.json"),
        f'{{"schema_version": "{SchemaVersion.ENVIRONMENT_SNAPSHOT}"}}',
    )

    assert findings == []
    assert families == {
        "sol_execbench.environment_snapshot": {
            SchemaVersion.ENVIRONMENT_SNAPSHOT,
        },
    }


def test_rejects_unregistered_schema_identifier():
    retired = "sol_execbench.agent_feedback." + "v2"

    findings, _ = audit_text(Path("example.json"), retired)

    assert findings == [
        f"example.json: unsupported schema identifier {retired}",
    ]


def test_accepts_current_solar_schema_identifier():
    findings, families = audit_text(
        Path("audit.json"),
        (
            '{"schema_version": '
            f'"{SolarSchemaVersion.RESOURCE_PEAK_CALIBRATION.value}"}}'
        ),
    )

    assert findings == []
    assert families == {
        "solar.resource_peak_calibration": {
            SolarSchemaVersion.RESOURCE_PEAK_CALIBRATION.value,
        },
    }


def test_accepts_current_non_namespaced_schema_identifier():
    version = SchemaVersion.RDNA4_DIAGNOSTIC_CORPUS_DESIGN.value

    findings, families = audit_text(
        Path("design.json"),
        f'{{"schema_version": "{version}"}}',
    )

    assert findings == []
    assert families == {"rdna4_diagnostic_corpus_design": {version}}


def test_accepts_current_numeric_schema_prose():
    findings, _ = audit_text(
        Path("guide.md"),
        "Both emit operator-graph schema v2.",
    )

    assert findings == []


def test_rejects_retired_numeric_schema_prose():
    retired = "operator-graph schema " + "v1"

    findings, _ = audit_text(Path("guide.md"), retired)

    assert findings == [
        "guide.md: unsupported operator_graph schema version 1; current is 2",
    ]


def test_rejects_non_current_tracked_numeric_artifact():
    findings, _ = audit_text(
        Path("problems/AMD_AKA/manifest.yaml"),
        "schema_version: 5\n",
    )

    assert findings == [
        (
            "problems/AMD_AKA/manifest.yaml: aka_corpus_manifest must contain "
            "only schema version 7"
        ),
    ]


def test_rejects_unregistered_numeric_schema_artifact():
    findings, _ = audit_text(
        Path("config/new-policy.yaml"),
        "schema_version: 1\n",
    )

    assert findings == [
        "config/new-policy.yaml: unregistered numeric schema artifact",
    ]


def test_rejects_raw_numeric_schema_in_production_python():
    findings, _ = audit_text(
        Path("src/example.py"),
        'payload = {"schema_version": 2}\n',
    )

    assert findings == [
        (
            "src/example.py:1: raw numeric schema version must use the "
            "matching family constant"
        ),
    ]


def test_rejects_raw_numeric_schema_in_all_python_contract_positions():
    samples = (
        "def load(schema_version: int = 2):\n    return schema_version\n",
        "load(schema_version=2)\n",
        'if payload.get("schema_version") != 2:\n    raise ValueError\n',
        'payload["schema_version"] = 2\n',
    )

    for content in samples:
        findings, _ = audit_text(Path("src/example.py"), content)
        assert findings == [
            (
                "src/example.py:1: raw numeric schema version must use the "
                "matching family constant"
            ),
        ]


def test_rejects_string_keyed_schema_registry_access():
    findings, _ = audit_text(
        Path("src/example.py"),
        'VERSION = SCHEMA_VERSIONS["example"]\n',
    )

    assert findings == [
        (
            "src/example.py:1: import the named schema-version enum member "
            "instead of indexing SCHEMA_VERSIONS"
        ),
    ]


def test_rejects_schema_enum_member_relay_constant():
    findings, _ = audit_text(
        Path("src/example.py"),
        "EXAMPLE_SCHEMA_VERSION = SchemaVersion.WORKLOAD\n",
    )

    assert findings == [
        (
            "src/example.py:1: use SchemaVersion.WORKLOAD directly instead "
            "of relaying it through EXAMPLE_SCHEMA_VERSION"
        ),
    ]


def test_rejects_schema_identifier_literal_outside_registry():
    findings, _ = audit_text(
        Path("src/example.py"),
        f'SCHEMA = "{SchemaVersion.WORKLOAD}"\n',
    )

    assert findings == [
        (
            f"src/example.py: schema identifier {SchemaVersion.WORKLOAD} "
            "must be referenced through SchemaVersion"
        ),
    ]


def test_rejects_schema_version_coercion_at_reader_boundaries():
    for conversion in ("int", "str"):
        findings, _ = audit_text(
            Path("src/example.py"),
            f'value = {conversion}(payload["schema_version"])\n',
        )

        assert findings == [
            (
                f"src/example.py:1: schema_version readers must not coerce "
                f"with {conversion}()"
            ),
        ]


def test_current_schema_registries_are_read_only():
    assert MODULE.registry_findings() == []


def test_rejects_retired_solar_schema_identifier():
    retired = "solar.verification.ir." + "v1"

    findings, _ = audit_text(Path("attestation.yaml"), retired)

    assert findings == [
        f"attestation.yaml: unsupported schema identifier {retired}",
    ]


def test_package_data_is_audited_but_root_data_is_excluded():
    assert not any(
        path == Path("src/sol_execbench/data")
        or path.is_relative_to("src/sol_execbench/data")
        for path in MODULE.EXCLUDED_PREFIXES
    )
    assert Path("data") in MODULE.EXCLUDED_PREFIXES


def test_upstream_tolerance_name_is_rejected_everywhere():
    upstream_name = "required_" + "match_ratio"

    findings, _ = audit_text(Path("public.json"), upstream_name)

    assert findings == [
        "public.json: upstream tolerance name escaped the import boundary",
    ]


def test_retired_workload_schema_migrator_stays_removed():
    assert (
        Path(
            "scripts/internal/migrate_workload_checks.py",
        )
        in MODULE.RETIRED_SCHEMA_PATHS
    )
    assert not (
        MODULE.ROOT / "scripts/internal/migrate_workload_checks.py"
    ).exists()

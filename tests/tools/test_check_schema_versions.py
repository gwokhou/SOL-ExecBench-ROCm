from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from sol_execbench.core.bench.performance_model.diagnostic_schema_versions import (
    DiagnosticArtifactSchema,
)
from sol_execbench.core.data.schema_versions import BenchmarkArtifactSchema
from sol_execbench.core.integrity.protocol_versions import WireProtocol
from sol_execbench.core.platform.schema_versions import PlatformArtifactSchema
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
        f'{{"schema_version": "{PlatformArtifactSchema.ENVIRONMENT_EVIDENCE}"}}',
    )

    assert findings == []
    assert families == {
        "sol_execbench.environment_evidence": {
            PlatformArtifactSchema.ENVIRONMENT_EVIDENCE,
        },
    }


def test_rejects_unregistered_schema_identifier():
    retired = "sol_execbench.agent_feedback." + "v2"

    findings, _ = audit_text(Path("example.json"), retired)

    assert findings == [
        f"example.json: unsupported versioned wire identifier {retired}",
    ]


def test_accepts_current_protocol_identifier():
    findings, families = audit_text(
        Path("trace.json"),
        f'{{"timing_protocol": "{WireProtocol.ROCM_EVENT_TIMING_CUSTOM}"}}',
    )

    assert findings == []
    assert families == {
        "sol_execbench.rocm_event_timing.custom": {
            WireProtocol.ROCM_EVENT_TIMING_CUSTOM,
        },
    }


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


def test_accepts_current_diagnostic_design_schema_identifier():
    version = DiagnosticArtifactSchema.DIAGNOSTIC_CORPUS_DESIGN.value

    findings, families = audit_text(
        Path("design.json"),
        f'{{"schema_version": "{version}"}}',
    )

    assert findings == []
    assert families == {"sol_execbench.diagnostic_corpus_design": {version}}


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


def test_accepts_local_cache_format_version():
    findings, _ = audit_text(
        Path("src/cache.py"),
        (
            "CACHE_FORMAT_VERSION = 1\n"
            'payload = {"format_version": CACHE_FORMAT_VERSION}\n'
        ),
    )

    assert findings == []


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
        "EXAMPLE_SCHEMA_VERSION = BenchmarkArtifactSchema.WORKLOAD\n",
    )

    assert findings == [
        (
            "src/example.py:1: use BenchmarkArtifactSchema.WORKLOAD directly instead "
            "of relaying it through EXAMPLE_SCHEMA_VERSION"
        ),
    ]


def test_rejects_schema_identifier_literal_outside_registry():
    findings, _ = audit_text(
        Path("src/example.py"),
        f'SCHEMA = "{BenchmarkArtifactSchema.WORKLOAD}"\n',
    )

    assert findings == [
        (
            f"src/example.py: versioned wire identifier "
            f"{BenchmarkArtifactSchema.WORKLOAD} must be referenced through "
            "BenchmarkArtifactSchema"
        ),
    ]


def test_rejects_protocol_identifier_literal_outside_registry():
    findings, _ = audit_text(
        Path("src/example.py"),
        f'PROTOCOL = "{WireProtocol.REFERENCE_IPC}"\n',
    )

    assert findings == [
        (
            f"src/example.py: versioned wire identifier "
            f"{WireProtocol.REFERENCE_IPC} must be referenced through "
            "WireProtocol"
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


def test_rejects_artifact_aggregate_import_in_production():
    findings, _ = audit_text(
        Path("src/example.py"),
        (
            "from sol_execbench.core.integrity.artifact_registry "
            "import CURRENT_STRING_ARTIFACT_SCHEMAS\n"
        ),
    )

    assert findings == [
        (
            "src/example.py:1: artifact_registry is audit-only; import the "
            "owning domain schema enum directly"
        ),
    ]


def test_current_schema_registries_are_read_only():
    assert MODULE.registry_findings() == []


def test_current_registries_have_non_test_owners():
    assert MODULE.registry_usage_findings(MODULE.first_party_paths()) == []


def test_rejects_retired_solar_schema_identifier():
    retired = "solar.verification.ir." + "v1"

    findings, _ = audit_text(Path("attestation.yaml"), retired)

    assert findings == [
        f"attestation.yaml: unsupported versioned wire identifier {retired}",
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

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

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
        '{"schema_version": "sol_execbench.environment_snapshot.v2"}',
    )

    assert findings == []
    assert families == {
        "sol_execbench.environment_snapshot": {
            "sol_execbench.environment_snapshot.v2",
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
        '{"schema_version": "solar.resource_peak_calibration.v3"}',
    )

    assert findings == []
    assert families == {
        "solar.resource_peak_calibration": {
            "solar.resource_peak_calibration.v3",
        },
    }


def test_rejects_retired_solar_schema_identifier():
    retired = "solar.verification.einsum." + "v1"

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

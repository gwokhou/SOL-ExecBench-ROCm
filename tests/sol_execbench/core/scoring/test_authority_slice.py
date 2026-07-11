from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from sol_execbench.cli.main import cli
from sol_execbench.core.scoring.amd_bound_sanity.builder import (
    build_amd_bound_sanity_report,
)
from sol_execbench.core.scoring.authority_slice import (
    AUTHORITY_SLICE_SCHEMA_VERSION,
    authority_slice_manifest_from_dict,
    build_authority_slice_manifest,
)


def _bound(uuid: str, *, status: str = "scored", confidence: str = "supported") -> dict:
    return {
        "schema_version": "sol_execbench.amd_sol_bound.v3",
        "definition": f"demo_{uuid}",
        "workload_uuid": uuid,
        "aggregate_bound": {"status": status},
        "hardware_model": {
            "architecture": "gfx1200",
            "hardware_validation_status": "validated",
            "model_validation_status": "validated",
        },
        "coverage_summary": {
            "inexact_ops": int(confidence == "inexact"),
            "worst_confidence": confidence,
        },
        "operator_work_estimates": [
            {
                "op_name": "matmul",
                "op_family": "gemm",
                "confidence": confidence,
            }
        ],
        "fusion_groups": [{"group_id": "group-0", "warnings": []}],
        "group_bounds": [
            {"group_id": "group-0", "confidence": "supported", "warnings": []}
        ],
        "warnings": [] if confidence == "supported" else ["inexact_operator:op_1:gemm"],
    }


def _solar(uuid: str, status: str = "scored") -> dict:
    return {
        "definition": f"demo_{uuid}",
        "workload_uuid": uuid,
        "aggregate_status": {"status": status, "warnings": []},
        "warnings": [],
    }


def _score(uuid: str) -> dict:
    return {
        "definition": f"demo_{uuid}",
        "workload_uuid": uuid,
        "supported": True,
        "evidence_refs": {
            "trace": f"trace/{uuid}",
            "timing": f"timing/{uuid}",
            "sol_bound": f"bound/{uuid}",
            "hardware_model": "hardware/gfx1200-v3.json",
        },
        "bound_eligibility": {
            "hardware_profile_state": "measured",
            "hardware_validation_status": "validated",
            "model_validation_status": "validated",
        },
        "warnings": [],
    }


def test_authority_slice_is_conservative_complete_and_deterministic() -> None:
    report = build_amd_bound_sanity_report(
        execution_closure={
            "records": [
                {"definition": "demo_w1", "workload_uuid": "w1"},
                {"definition": "demo_w2", "workload_uuid": "w2"},
            ]
        },
        amd_sol_artifacts=[_bound("w1"), _bound("w2", confidence="inexact")],
        solar_artifacts=[_solar("w1"), _solar("w2")],
        amd_score_report={"scores": [_score("w1"), _score("w2")]},
        compatibility_matrix={},
        created_at="2026-07-11T00:00:00Z",
    )
    suite = [
        {"definition": "demo_w2", "workload_uuid": "w2"},
        {"definition": "demo_w1", "workload_uuid": "w1"},
        {"definition": "demo_w3", "workload_uuid": "w3"},
    ]

    first = build_authority_slice_manifest(
        suite_workloads=suite,
        source_suite_manifest_sha256="a" * 64,
        sanity_report=report,
    )
    second = build_authority_slice_manifest(
        suite_workloads=list(reversed(suite)),
        source_suite_manifest_sha256="a" * 64,
        sanity_report=report,
    )

    assert first.schema_version == AUTHORITY_SLICE_SCHEMA_VERSION
    assert [row.workload_uuid for row in first.workloads] == ["w1"]
    assert first.source_workload_count == 3
    assert {row.workload_uuid for row in first.excluded} == {"w2", "w3"}
    assert "inexact_operator" in next(
        row.blocker_codes for row in first.excluded if row.workload_uuid == "w2"
    )
    assert "missing_authority_audit" in next(
        row.blocker_codes for row in first.excluded if row.workload_uuid == "w3"
    )
    assert first.to_json() == second.to_json()
    assert authority_slice_manifest_from_dict(json.loads(first.to_json())) == first


def test_authority_slice_rejects_digest_tampering() -> None:
    report = build_amd_bound_sanity_report(created_at="2026-07-11T00:00:00Z")
    manifest = build_authority_slice_manifest(
        suite_workloads=[],
        source_suite_manifest_sha256="b" * 64,
        sanity_report=report,
    )
    payload = json.loads(manifest.to_json())
    payload["source_suite_manifest_sha256"] = "c" * 64

    with pytest.raises(ValueError, match="checksum mismatch"):
        authority_slice_manifest_from_dict(payload)


def test_authority_slice_does_not_trust_a_pre_audit_sanity_report() -> None:
    report = build_amd_bound_sanity_report(
        execution_closure={
            "records": [{"definition": "demo_w1", "workload_uuid": "w1"}]
        },
        amd_sol_artifacts=[_bound("w1")],
        solar_artifacts=[_solar("w1")],
        amd_score_report={"scores": [_score("w1")]},
        compatibility_matrix={},
        created_at="2026-07-11T00:00:00Z",
    ).model_copy(update={"authority_audit_policy_version": None})

    manifest = build_authority_slice_manifest(
        suite_workloads=[{"definition": "demo_w1", "workload_uuid": "w1"}],
        source_suite_manifest_sha256="d" * 64,
        sanity_report=report,
    )

    assert manifest.workloads == ()
    assert manifest.excluded[0].blocker_codes == ("authority_audit_policy_mismatch",)


def test_sanity_audit_groups_operator_family_and_blocker_codes() -> None:
    report = build_amd_bound_sanity_report(
        amd_sol_artifacts=[_bound("w2", confidence="inexact")],
        created_at="2026-07-11T00:00:00Z",
    )

    assert report.operator_counts == {"matmul": 1}
    assert report.op_family_counts == {"gemm": 1}
    assert report.blocker_counts_by_operator == {"matmul": {"inexact_operator": 1}}
    assert report.blocker_counts_by_op_family == {"gemm": {"inexact_operator": 1}}
    assert "inexact_operator" in report.workloads[0].blocker_codes


def test_authority_freeze_cli_writes_a_baseline_compatible_manifest(tmp_path) -> None:
    report = build_amd_bound_sanity_report(
        execution_closure={
            "records": [{"definition": "demo_w1", "workload_uuid": "w1"}]
        },
        amd_sol_artifacts=[_bound("w1")],
        solar_artifacts=[_solar("w1")],
        amd_score_report={"scores": [_score("w1")]},
        compatibility_matrix={},
        created_at="2026-07-11T00:00:00Z",
    )
    suite_path = tmp_path / "suite.json"
    sanity_path = tmp_path / "sanity.json"
    output_path = tmp_path / "authority-slice.json"
    suite_path.write_text(
        json.dumps({"workloads": [{"definition": "demo_w1", "workload_uuid": "w1"}]}),
        encoding="utf-8",
    )
    sanity_path.write_text(report.to_json(), encoding="utf-8")

    result = CliRunner().invoke(
        cli,
        [
            "baseline",
            "authority",
            "freeze",
            "--suite-manifest",
            str(suite_path),
            "--sanity-report",
            str(sanity_path),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == AUTHORITY_SLICE_SCHEMA_VERSION
    assert payload["workloads"] == [
        {
            "blocker_codes": [],
            "category": "unknown",
            "definition": "demo_w1",
            "evidence_refs": {
                "hardware_model": "hardware/gfx1200-v3.json",
                "sol_bound": "bound/w1",
                "timing": "timing/w1",
                "trace": "trace/w1",
            },
            "prerequisite_evidence_summary": {
                "amd_sol_status": "scored",
                "diagnostic_flags": ["scored"],
                "diagnostic_status": "scored",
                "solar_status": "scored",
            },
            "problem_id": "demo_w1",
            "row_index": None,
            "workload_uuid": "w1",
        }
    ]

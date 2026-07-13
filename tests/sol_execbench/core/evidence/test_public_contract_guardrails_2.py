from __future__ import annotations

import json

from pathlib import Path

import pytest

from click.testing import CliRunner

from sol_execbench.cli.main import cli

from sol_execbench.core.bench.rocm_profiler import (
    Rocprofv3TimingEvidence,
    Rocprofv3TimingRow,
)

from sol_execbench.core.bench.timing_policy import (
    TimingActivityDomain,
    TimingBackend,
    TimingSourceType,
    select_timing_policy,
)

from sol_execbench.core.data.trace import Trace

from sol_execbench.core.data.workload import Workload


from sol_execbench.core.scoring.amd_score import (
    DEGRADED_SOL_BOUND_WARNING,
    UNSCORED_SOL_BOUND_WARNING,
    build_amd_native_suite_report,
    score_amd_native_workload,
)


from sol_execbench.core.data.definition import Definition

from sol_execbench.core.scoring import solar_derivation as solar_derivation_module

from sol_execbench.core.scoring.solar_derivation import SolarAggregateStatus

from sol_execbench_type_helpers import (
    make_amd_hardware_model,
    make_amd_sol_bound,
    make_definition,
    make_trace,
    make_workload,
)


REPO_ROOT = Path(__file__).resolve().parents[4]

PLANNING_ROOT = REPO_ROOT / ".planning"

COMPATIBILITY_INVENTORY = REPO_ROOT / "docs/internal/v1_4_compatibility_inventory.md"

PHASE50_INTERNAL_EVIDENCE_NAMES = (
    "moe_static_route_flops",
    "moe_dynamic_route_bytes",
    "ssm_mamba_static_scan_flops",
    "ssm_mamba_degraded_scan_bytes",
    "inexact_operator:moe_dynamic_routing",
    "unsupported_operator:moe_taxonomy_only",
    "inexact_operator:ssm_missing_recurrence",
    "unsupported_operator:ssm_custom_scan",
    "aggregate_degraded:moe",
    "aggregate_degraded:ssm_mamba",
    "aggregate_unscored:moe",
    "aggregate_unscored:ssm_mamba",
    "route:top_k",
    "route:static_cardinality",
    "recurrence:state_shape",
    "recurrence:update_parameters",
)

PHASE51_SCORE_INTERNAL_EVIDENCE_REFS = (
    "solar_derivation",
    "coverage_summary",
    "aggregate_status",
    "family_counts",
    "status_counts",
    "degraded_node_ids",
    "unsupported_node_ids",
    "estimated_node_ids",
    "score_eligible",
    "formula_evidence",
    "byte_evidence",
    "bound_evidence",
)

PHASE51_INTERNAL_PUBLIC_BOUNDARY_FIELDS = (
    *PHASE51_SCORE_INTERNAL_EVIDENCE_REFS,
    "missing_patterns",
    "unsupported_patterns",
    "provenance",
    "source_boundary",
    "candidate_solution_execution",
)

PHASE51_SOLAR_ONLY_ARTIFACT_FIELDS = (
    "solar_derivation",
    "aggregate_status",
    "family_counts",
    "status_counts",
    "degraded_node_ids",
    "unsupported_node_ids",
    "estimated_node_ids",
    "score_eligible",
    "formula_evidence",
    "byte_evidence",
    "bound_evidence",
)

CANONICAL_DEFINITION_KEYS = {
    "name",
    "op_type",
    "axes",
    "custom_inputs_entrypoint",
    "inputs",
    "outputs",
    "reference",
    "description",
    "hf_id",
}

CANONICAL_WORKLOAD_KEYS = {"axes", "inputs", "uuid", "tolerance"}

CANONICAL_TRACE_KEYS = {"definition", "workload", "solution", "evaluation"}

PUBLIC_SCORE_EVIDENCE_REF_KEYS = {
    "trace",
    "timing",
    "sol_bound",
    "baseline",
    "hardware_model",
}

DERIVED_REPORT_EVIDENCE_REF_KEYS = {
    "formula",
    "hardware_model",
    "coverage",
    "score_eligibility",
}


def _json_object_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        keys = {str(key) for key in value}
        for nested in value.values():
            keys.update(_json_object_keys(nested))
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for nested in value:
            keys.update(_json_object_keys(nested))
        return keys
    return set()


def _sample_definition_workload_trace() -> tuple[Definition, Workload, Trace]:
    definition = make_definition(
        name="demo",
        axes={"N": {"type": "var"}},
        inputs={"x": {"shape": ["N"], "dtype": "float32"}},
        outputs={"out": {"shape": ["N"], "dtype": "float32"}},
        reference="def run(x):\n    return x",
    )
    workload = make_workload(
        axes={"N": 16}, inputs={"x": {"type": "random"}}, uuid="w1"
    )
    trace = make_trace(
        definition="demo",
        workload=workload,
        solution="solution",
        evaluation=None,
    )
    return definition, workload, trace


def test_primary_cli_does_not_expose_v1_19_amd_bound_sanity_options():
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    help_text = result.output

    for option in (
        "--amd-bound-sanity",
        "--report-amd-bound-sanity",
        "--amd-sol-artifact",
        "--solar-artifact",
        "--compatibility-matrix",
        "--created-at",
        "amd-bound-sanity",
        "report_amd_bound_sanity",
    ):
        assert option not in help_text


def test_v1_20_consistency_report_fields_remain_sidecar_only():
    definition, workload, trace = _sample_definition_workload_trace()
    forbidden = (
        "sol_execbench.consistency_report.v1",
        "consistency_report",
        "consistency_report_checksum",
        "ConsistencyReport",
        "ConsistencyFinding",
        "sources_checked",
        "finding_totals",
        "denominator_closure_drift",
        "matrix_runtime_unavailable_attempted",
        "missing_derived_evidence_scored",
        "source_ref_checksum_mismatch",
        "claim_boundary_violation",
        "diagnostic_only",
        "score_authority",
        "paper_parity",
        "leaderboard_authority",
        "native_host_validation",
        "new_hardware_validation",
    )

    for payload in (
        definition.model_dump(mode="json"),
        workload.model_dump(mode="json"),
        trace.model_dump(mode="json"),
    ):
        text = json.dumps(payload, sort_keys=True)
        for field in forbidden:
            assert field not in text


def test_primary_cli_does_not_expose_v1_20_consistency_options():
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    help_text = result.output

    for option in (
        "--consistency-report",
        "--report-consistency",
        "--paper-denominator",
        "--matrix-report",
        "--created-at",
        "report_consistency",
        "consistency-report",
    ):
        assert option not in help_text


def test_v1_20_evaluation_stability_fields_remain_sidecar_only():
    definition, workload, trace = _sample_definition_workload_trace()
    forbidden = (
        "sol_execbench.evaluation_stability.v1",
        "evaluation_stability",
        "evaluation_stability_checksum",
        "EvaluationStabilityReport",
        "StabilityWorkload",
        "stability_status",
        "runtime_distribution",
        "coefficient_of_variation",
        "selected_statistic",
        "clock_unlocked",
        "profiler_overhead_risk",
        "backend_unsupported",
        "timing_quality_interpretation",
        "correctness_authority",
        "score_authority",
        "paper_parity",
        "leaderboard_authority",
        "native_host_validation",
        "new_hardware_validation",
    )

    for payload in (
        definition.model_dump(mode="json"),
        workload.model_dump(mode="json"),
        trace.model_dump(mode="json"),
    ):
        text = json.dumps(payload, sort_keys=True)
        for field in forbidden:
            assert field not in text


def test_primary_cli_does_not_expose_v1_20_stability_options():
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    help_text = result.output

    for option in (
        "--evaluation-stability",
        "--report-evaluation-stability",
        "--noise-cv-threshold",
        "--min-samples",
        "report_evaluation_stability",
        "evaluation-stability",
    ):
        assert option not in help_text


def test_v1_20_claim_upgrade_fields_remain_sidecar_only():
    definition, workload, trace = _sample_definition_workload_trace()
    forbidden = (
        "sol_execbench.claim_upgrade.v1",
        "claim_upgrade",
        "claim_upgrade_checksum",
        "ClaimUpgradeReport",
        "ClaimEvaluation",
        "highest_eligible_claim",
        "diagnostic_only",
        "container_validated",
        "native_host_validated",
        "score_authoritative",
        "paper_parity_candidate",
        "leaderboard_ready",
        "unmet_prerequisites",
        "prerequisite_evaluation_only",
        "mutates_source_authority",
    )

    for payload in (
        definition.model_dump(mode="json"),
        workload.model_dump(mode="json"),
        trace.model_dump(mode="json"),
    ):
        text = json.dumps(payload, sort_keys=True)
        for field in forbidden:
            assert field not in text


def test_primary_cli_does_not_expose_v1_20_claim_upgrade_options():
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    help_text = result.output

    for option in (
        "--claim-upgrade",
        "--report-claim-upgrade",
        "--hardware-validation",
        "report_claim_upgrade",
        "claim-upgrade",
    ):
        assert option not in help_text


def test_v1_20_trust_summary_fields_remain_sidecar_only():
    definition, workload, trace = _sample_definition_workload_trace()
    forbidden = (
        "sol_execbench.trust_summary.v1",
        "trust_summary",
        "trust_summary_checksum",
        "TrustSummaryReport",
        "TrustOutcome",
        "overall_status",
        "internally_consistent",
        "stable_enough_to_interpret",
        "evidence_completeness",
        "claim_upgrade_blocked",
        "review_guidance_only",
        "paper_validation",
    )

    for payload in (
        definition.model_dump(mode="json"),
        workload.model_dump(mode="json"),
        trace.model_dump(mode="json"),
    ):
        text = json.dumps(payload, sort_keys=True)
        for field in forbidden:
            assert field not in text


def test_primary_cli_does_not_expose_v1_20_trust_summary_options():
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    help_text = result.output

    for option in (
        "--trust-summary",
        "--report-trust-summary",
        "--claim-upgrade",
        "report_trust_summary",
        "trust-summary",
    ):
        assert option not in help_text


def test_v1_19_amd_bound_sanity_markdown_keeps_negative_boundaries_visible():
    from sol_execbench.core.scoring.amd_bound_sanity.builder import (
        build_amd_bound_sanity_report,
    )
    from sol_execbench.core.scoring.amd_bound_sanity.rendering import (
        render_amd_bound_sanity_markdown,
    )

    report = build_amd_bound_sanity_report(
        execution_closure={
            "schema_version": "sol_execbench.execution_closure.v1",
            "records": [
                {
                    "category": "L1",
                    "problem_id": "L1/demo",
                    "workload_uuid": "w1",
                    "closure_status": "derived_evidence_missing",
                    "evidence_refs": {},
                    "evidence_gaps": ["amd_sol_evidence_missing"],
                }
            ],
        },
        created_at="2026-05-31T00:00:00Z",
    )
    markdown = render_amd_bound_sanity_markdown(report)

    for expected in (
        "diagnostic existing evidence sanity report",
        "not upstream SOLAR equivalence",
        "not AMD SOL/SOLAR model validation",
        "not paper parity",
        "not leaderboard authority",
        "not score authority upgrade",
        "not CDNA 3 validation",
        "not MI300X validation",
        "not CDNA 4 validation",
        "not native-host validation",
        "not new-hardware validation",
        "`upstream_solar_equivalence`: false",
        "`score_authority_upgrade`: false",
    ):
        assert expected in markdown


def test_phase88_example_docs_keep_v1_19_surfaces_sidecar_only():
    definition, workload, trace = _sample_definition_workload_trace()
    examples_readme = (REPO_ROOT / "docs/examples/v1_19_evidence/README.md").read_text()

    for expected in (
        "demo-only",
        "diagnostic-only",
        "sidecars/reports only",
        "canonical Trace, Definition, Workload, Solution",
        "no score authority",
        "no leaderboard readiness",
        "no native-host ROCm Matrix validation",
    ):
        assert expected in examples_readme

    for payload in (
        definition.model_dump(mode="json"),
        workload.model_dump(mode="json"),
        trace.model_dump(mode="json"),
    ):
        text = json.dumps(payload, sort_keys=True)
        for field in (
            "sol_execbench.execution_closure.v1",
            "sol_execbench.paper_denominator_report.v1",
            "sol_execbench.rocm_compatibility_matrix_diff.v1",
            "sol_execbench.amd_bound_sanity.v1",
            "demo-only",
            "diagnostic-only",
            "paper_denominator",
            "amd_bound_sanity",
            "matrix_diff",
        ):
            assert field not in text


def test_v1_11_parity_gap_docs_keep_bounded_claim_boundary():
    docs = (REPO_ROOT / "docs/internal/analysis.md").read_text()

    assert "scripts/internal/reports/report_parity_gaps.py" in docs
    assert "discovered, parsed, ready, blocked" in docs
    assert "not full validation" in docs
    assert "not paper parity" in docs
    assert "not upstream SOLAR parity" in docs
    assert "not NVIDIA B200 or" in docs
    assert "not hosted leaderboard readiness" in docs


def test_v1_11_release_closure_summarizes_remaining_claim_gaps():
    doc = (REPO_ROOT / "docs/internal/v1_11_release_closure.md").read_text()

    for expected in (
        "not full 235-problem ROCm validation",
        "not original 124-model",
        "not upstream SOLAR parity",
        "not NVIDIA B200 or Blackwell equivalence",
        "not hosted leaderboard readiness",
        "MI300X / CDNA 3 full-suite validation",
        "CDNA 4 validation",
        "NVFP4 and MXFP4 validation",
        "sidecar-only",
    ):
        assert expected in doc


def test_primary_cli_does_not_expose_v1_11_dataset_inspection_options():
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    help_text = result.output

    for option in (
        "--inventory",
        "--readiness",
        "--ready-subset",
        "--execution-closure",
        "--dataset-manifest",
        "--dataset-root",
    ):
        assert option not in help_text


def test_public_score_evidence_refs_keep_exact_established_key_space():
    definition = make_definition(
        name="matmul_demo",
        axes={
            "M": {"type": "var"},
            "K": {"type": "const", "value": 4},
            "N": {"type": "const", "value": 8},
        },
        inputs={
            "a": {"shape": ["M", "K"], "dtype": "float32"},
            "b": {"shape": ["K", "N"], "dtype": "float32"},
        },
        outputs={"out": {"shape": ["M", "N"], "dtype": "float32"}},
        reference="def run(a, b):\n    return a @ b",
    )
    workload = make_workload(
        axes={"M": 2},
        inputs={"a": {"type": "random"}, "b": {"type": "random"}},
        uuid="matmul-workload",
    )
    artifact = make_amd_sol_bound(
        definition,
        workload,
        make_amd_hardware_model(),
    )

    payload = score_amd_native_workload(
        artifact,
        measured_latency_ms=1.0,
        baseline_latency_ms=2.0,
        trace_ref="traces/matmul.json",
        timing_evidence_ref="timing/matmul.json",
        sol_bound_ref="bounds/matmul.json",
        baseline_ref="baseline/matmul.json",
        hardware_model_ref="hardware/gfx1200.json",
        derived_evidence_refs={
            "formula": "solar/matmul.json#groups.formula_evidence",
            "hardware_model": "hardware/gfx1200.json",
            "coverage": "solar/matmul.json#coverage_summary",
            "score_eligibility": "solar/matmul.json#aggregate_status",
        },
    ).to_dict()

    assert set(payload["evidence_refs"]) == PUBLIC_SCORE_EVIDENCE_REF_KEYS
    assert set(payload["derived_evidence_refs"]) == DERIVED_REPORT_EVIDENCE_REF_KEYS
    assert set(payload["evidence_refs"]).isdisjoint(
        {"formula", "coverage", "score_eligibility"}
    )


def test_complex_family_without_export_semantic_proof_is_unscored():
    cases = (
        (
            make_definition(
                name="moe_dynamic_route",
                axes={
                    "tokens": {"type": "const", "value": 128},
                    "hidden": {"type": "const", "value": 256},
                    "experts": {"type": "const", "value": 8},
                },
                inputs={
                    "x": {"shape": ["tokens", "hidden"], "dtype": "float16"},
                    "router": {"shape": ["hidden", "experts"], "dtype": "float16"},
                    "expert_weights": {
                        "shape": ["experts", "hidden", "hidden"],
                        "dtype": "float16",
                    },
                    "threshold": {"shape": None, "dtype": "float16"},
                },
                outputs={"out": {"shape": ["tokens", "hidden"], "dtype": "float16"}},
                reference=(
                    "def run(x, router, expert_weights, threshold):\n"
                    "    scores = router(x)\n"
                    "    chosen = scores > threshold\n"
                    "    return dispatch_dynamic(x, expert_weights, chosen)\n"
                ),
            ),
            make_workload(
                axes={},
                inputs={
                    "x": {"type": "random"},
                    "router": {"type": "random"},
                    "expert_weights": {"type": "random"},
                    "threshold": {"type": "random"},
                },
                uuid="moe-dynamic-workload",
            ),
            "moe_dynamic_route_bytes",
        ),
        (
            make_definition(
                name="ssm_mamba_missing_recurrence",
                axes={
                    "batch": {"type": "const", "value": 2},
                    "sequence": {"type": "const", "value": 64},
                    "hidden": {"type": "const", "value": 128},
                    "one": {"type": "const", "value": 1},
                    "kernel": {"type": "const", "value": 3},
                },
                inputs={
                    "x": {"shape": ["batch", "sequence", "hidden"], "dtype": "float16"},
                    "w_in": {"shape": ["hidden", "hidden"], "dtype": "float16"},
                    "conv_weight": {
                        "shape": ["hidden", "one", "kernel"],
                        "dtype": "float16",
                    },
                    "params": {"shape": ["hidden"], "dtype": "float16"},
                    "w_out": {"shape": ["hidden", "hidden"], "dtype": "float16"},
                },
                outputs={
                    "out": {
                        "shape": ["batch", "sequence", "hidden"],
                        "dtype": "float16",
                    }
                },
                reference=(
                    "def run(x, w_in, conv_weight, params, w_out):\n"
                    "    z = in_proj(x, w_in)\n"
                    "    z = depthwise_conv(z, conv_weight)\n"
                    "    y = selective_scan(z, params)\n"
                    "    return out_proj(y, w_out)\n"
                ),
            ),
            make_workload(
                axes={},
                inputs={
                    "x": {"type": "random"},
                    "w_in": {"type": "random"},
                    "conv_weight": {"type": "random"},
                    "params": {"type": "random"},
                    "w_out": {"type": "random"},
                },
                uuid="ssm-mamba-workload",
            ),
            "ssm_mamba_degraded_scan_bytes",
        ),
    )
    hardware = make_amd_hardware_model()

    for definition, workload, formula_kind in cases:
        artifact = make_amd_sol_bound(definition, workload, hardware)
        score = score_amd_native_workload(
            artifact,
            measured_latency_ms=1.0,
            baseline_latency_ms=2.0,
            hardware_model_ref="hardware/gfx1200.json",
        )

        assert artifact.aggregate_bound.status == "unscored"
        assert any(
            estimate["formula_kind"] == formula_kind
            for estimate in artifact.operator_work_estimates
        )
        assert score.supported is False
        assert score.claim_level == "amd-native-derived"
        assert UNSCORED_SOL_BOUND_WARNING in score.warnings
        for field in PHASE51_SCORE_INTERNAL_EVIDENCE_REFS:
            assert field not in score.evidence_refs


def test_importing_solar_derivation_keeps_amd_native_score_eligibility_unchanged():
    assert (
        solar_derivation_module.SOLAR_DERIVATION_SCHEMA_VERSION
        == "sol_execbench.solar_derivation.v1"
    )

    definition = make_definition(
        name="matmul_demo",
        axes={
            "M": {"type": "var"},
            "K": {"type": "const", "value": 4},
            "N": {"type": "const", "value": 8},
        },
        inputs={
            "a": {"shape": ["M", "K"], "dtype": "float32"},
            "b": {"shape": ["K", "N"], "dtype": "float32"},
        },
        outputs={"out": {"shape": ["M", "N"], "dtype": "float32"}},
        reference="def run(a, b):\n    return a @ b",
    )
    workload = make_workload(
        axes={"M": 2},
        inputs={"a": {"type": "random"}, "b": {"type": "random"}},
        uuid="matmul-workload",
    )
    hardware = make_amd_hardware_model()
    v3_artifact = make_amd_sol_bound(
        definition,
        workload,
        hardware,
        hardware_model_ref="hardware/gfx1200.json",
    )

    v3_score = score_amd_native_workload(
        v3_artifact,
        measured_latency_ms=1.0,
        baseline_latency_ms=2.0,
        hardware_model_ref="hardware/gfx1200.json",
    )

    assert v3_score.supported is True
    assert v3_score.to_dict()["claim_level"] == "amd-native-derived"
    for field in PHASE51_SCORE_INTERNAL_EVIDENCE_REFS:
        assert field not in v3_score.to_dict()["evidence_refs"]
    v3_artifact_keys = _json_object_keys(v3_artifact.to_dict())
    for field in PHASE51_SOLAR_ONLY_ARTIFACT_FIELDS:
        assert field not in v3_artifact_keys


def test_solar_score_guard_does_not_expose_internal_evidence_refs_or_claims():
    definition = make_definition(
        name="matmul_demo",
        axes={
            "M": {"type": "var"},
            "K": {"type": "const", "value": 4},
            "N": {"type": "const", "value": 8},
        },
        inputs={
            "a": {"shape": ["M", "K"], "dtype": "float32"},
            "b": {"shape": ["K", "N"], "dtype": "float32"},
        },
        outputs={"out": {"shape": ["M", "N"], "dtype": "float32"}},
        reference="def run(a, b):\n    return a @ b",
    )
    workload = make_workload(
        axes={"M": 2},
        inputs={"a": {"type": "random"}, "b": {"type": "random"}},
        uuid="matmul-workload",
    )
    artifact = make_amd_sol_bound(
        definition,
        workload,
        make_amd_hardware_model(),
    )

    unscored = score_amd_native_workload(
        artifact,
        measured_latency_ms=1.0,
        baseline_latency_ms=2.0,
        timing_evidence_ref="timing.json",
        sol_bound_ref="sol.json",
        solar_derivation=SolarAggregateStatus(
            status="unscored",
            score_eligible=False,
            reason="test unscored aggregate",
            group_ids=("group-1",),
            node_ids=("node-1",),
            warnings=("aggregate_unscored:unsupported semantic evidence",),
        ),
    ).to_dict()
    degraded = score_amd_native_workload(
        artifact,
        measured_latency_ms=1.0,
        baseline_latency_ms=2.0,
        timing_evidence_ref="timing.json",
        sol_bound_ref="sol.json",
        solar_derivation=SolarAggregateStatus(
            status="degraded",
            score_eligible=True,
            reason="test degraded aggregate",
            group_ids=("group-1",),
            node_ids=("node-1",),
            warnings=("aggregate_degraded:incomplete semantic evidence",),
        ),
    ).to_dict()

    assert unscored["claim_level"] == "amd-native-derived"
    assert degraded["claim_level"] == "amd-native-derived"
    assert UNSCORED_SOL_BOUND_WARNING in unscored["warnings"]
    assert DEGRADED_SOL_BOUND_WARNING in degraded["warnings"]
    for payload in (unscored, degraded):
        for field in PHASE51_SCORE_INTERNAL_EVIDENCE_REFS:
            assert field not in payload["evidence_refs"]


def test_v1_9_derived_artifacts_remain_noncanonical():
    policy = select_timing_policy(TimingSourceType.HIP_NATIVE)
    timing = Rocprofv3TimingEvidence(
        tool_version="rocprofv3 7.0.0",
        gpu_architecture="gfx1200",
        activity_domain=TimingActivityDomain.KERNEL_ACTIVITY,
        aggregation_rule=policy.aggregation_rule,
        backend=TimingBackend.ROCPROFV3,
        interpretation=policy.interpretation,
        parsed_rows=(
            Rocprofv3TimingRow(
                name="kernel",
                domain="KERNEL_DISPATCH",
                duration_ns=1000.0,
            ),
        ),
    )
    suite = build_amd_native_suite_report([])

    assert timing.to_dict()["derived"] is True
    assert timing.to_dict()["canonical_output"] == "trace_jsonl"
    assert suite.to_dict()["derived"] is True
    assert suite.to_dict()["canonical_output"] == "trace_jsonl"


def test_v1_9_claim_guardrails_keep_cdna3_and_nvidia_equivalence_out_of_scope():
    if (
        not (PLANNING_ROOT / "PROJECT.md").exists()
        or not (PLANNING_ROOT / "REQUIREMENTS.md").exists()
    ):
        pytest.skip("SOL planning metadata is not present in this nested checkout")
    project = (PLANNING_ROOT / "PROJECT.md").read_text()
    requirements = (PLANNING_ROOT / "REQUIREMENTS.md").read_text()
    analysis = Path("docs/internal/analysis.md").read_text()

    assert "CDNA 3 (`gfx94*`) full adapted suite validation remains deferred" in project
    assert (
        "CDNA3 or CDNA4 validation claim upgrade" in requirements
        or "CDNA3-family" in requirements
    )
    assert (
        "MI300X and MI308X are sibling GPU products" in requirements
        or "CDNA3 or CDNA4 validation claim upgrade" in requirements
    )
    assert "not NVIDIA B200, SOLAR, or leaderboard equivalence claims" in analysis
    assert "hardware_validation_status" in analysis
    assert "model_validation_status" in analysis

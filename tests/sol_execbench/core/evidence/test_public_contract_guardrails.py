from __future__ import annotations

import json

from pathlib import Path


from click.testing import CliRunner

from sol_execbench.cli.main import cli


from sol_execbench.core.data.trace import Trace

from sol_execbench.core.data.workload import Workload

from sol_execbench.core.dataset import DatasetManifestSource, build_dataset_manifest

from sol_execbench.core.scoring.amd_score import (
    build_amd_native_suite_report,
)


from sol_execbench.core.data.definition import Definition


from sol_execbench_type_helpers import (
    json_dict,
    make_amd_hardware_model,
    make_amd_sol_bound,
    make_definition,
    make_solution,
    make_trace,
    make_workload,
)

from ..scoring.solar_derivation_fixtures import load_solar_derivation_fixtures

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


def test_solution_json_contract_accepts_existing_rocm_shape():
    solution = make_solution(
        name="demo",
        definition="demo_problem",
        author="tester",
        spec={
            "languages": ["hip_cpp"],
            "target_hardware": ["gfx1200", "gfx942"],
            "entry_point": "kernel.hip::run",
            "compile_options": {"hip_cflags": ["-O3"]},
        },
        sources=[
            {"path": "kernel.hip", "content": 'extern "C" __global__ void k() {}'}
        ],
    )

    dumped = solution.model_dump(mode="json")
    assert dumped["spec"]["target_hardware"] == ["gfx1200", "gfx942"]
    assert dumped["spec"]["compile_options"]["hip_cflags"] == ["-O3"]


def test_workload_jsonl_contract_keeps_uuid_and_input_shape():
    raw = {"axes": {"n": 16}, "inputs": {"x": {"type": "random"}}, "uuid": "w1"}
    workload = make_workload(**raw)
    assert workload.model_dump(mode="json")["uuid"] == "w1"
    assert workload.model_dump(mode="json")["inputs"]["x"]["type"] == "random"


def test_trace_jsonl_contract_accepts_existing_workload_only_trace():
    raw = {
        "definition": "demo",
        "workload": {"axes": {}, "inputs": {}, "uuid": "w1"},
        "solution": None,
        "evaluation": None,
    }
    trace = make_trace(**raw)
    assert trace.is_workload_trace()
    dumped = trace.model_dump(mode="json")
    assert dumped["definition"] == raw["definition"]
    assert dumped["solution"] is None
    assert dumped["evaluation"] is None
    assert dumped["workload"]["uuid"] == "w1"


def test_canonical_definition_workload_trace_top_level_keys_are_exact():
    definition, workload, trace = _sample_definition_workload_trace()

    assert set(definition.model_dump(mode="json")) == CANONICAL_DEFINITION_KEYS
    assert set(workload.model_dump(mode="json")) == CANONICAL_WORKLOAD_KEYS
    assert set(trace.model_dump(mode="json")) == CANONICAL_TRACE_KEYS


def test_canonical_trace_jsonl_excludes_derived_report_key_space():
    _, _, trace = _sample_definition_workload_trace()
    payload = trace.model_dump(mode="json")
    serialized = json.dumps(payload, sort_keys=True)
    forbidden_keys = {
        "baseline_export_fields",
        "capabilities",
        "compatibility_metadata_fields",
        "contract_version",
        "derived_evidence_refs",
        "failure_categories",
        "formula",
        "coverage",
        "runtime.evidence.v1",
        "environment_snapshot",
        "static_kernel_evidence",
        "static_kernel_evidence.v2",
        "sol_execbench.static_kernel_evidence.v3",
        "diagnostic_only",
        "correctness_authority",
        "performance_authority",
        "timing_authority",
        "score_authority",
        "paper_parity_authority",
        "leaderboard_authority",
        "metadata_present",
        "disassembly_present",
        "detected_architectures",
        "symbol_count",
        "score_eligibility",
        *DERIVED_REPORT_EVIDENCE_REF_KEYS,
        *PHASE51_INTERNAL_PUBLIC_BOUNDARY_FIELDS,
    }

    assert _json_object_keys(payload).isdisjoint(forbidden_keys)
    for key in forbidden_keys:
        assert key not in serialized


def test_v1_18_compatibility_matrix_fields_remain_sidecar_only():
    definition, workload, trace = _sample_definition_workload_trace()
    forbidden = (
        "sol_execbench.rocm_compatibility_matrix.v1",
        "MatrixEntry",
        "target",
        "observed",
        "diagnostic_compatibility_evidence",
        "container_user_space_validated",
        "native_host_validated",
        "MatrixTarget",
        "MatrixObservedEvidence",
        "requested_rocm_user_space_version",
        "docker_image_repository",
        "docker_image_tag",
        "pytorch_rocm_target",
        "validation_scope",
        "host_validated",
        "container_validated",
        "mixed_version",
        "pytorch_wheel_unavailable",
        "runtime_unavailable",
        "not_tested",
    )

    for payload in (
        definition.model_dump(mode="json"),
        workload.model_dump(mode="json"),
        trace.model_dump(mode="json"),
    ):
        assert _json_object_keys(payload).isdisjoint(forbidden)
        text = json.dumps(payload, sort_keys=True)
        for field in forbidden:
            assert field not in text


def test_cli_help_exposes_v2_tree_and_evaluate_options():
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    help_text = result.output
    for command in (
        "evaluate",
        "environment",
        "contract",
        "toolchain",
        "dataset",
        "baseline",
        "hardware",
        "score",
    ):
        assert command in help_text
    evaluate_help = CliRunner().invoke(cli, ["evaluate", "--help"]).output
    for expected_option in (
        "Usage:",
        "--definition",
        "--workload",
        "--solution",
        "--config",
        "--compile-timeout",
        "--timeout",
        "--trace-output",
        "--lock-clocks",
        "--keep-staging",
        "--profile",
        "--static-evidence",
        "--verbose",
    ):
        assert expected_option in evaluate_help
    assert "--json" not in evaluate_help
    for unexpected_option in (
        "diagnose",
        "--rocprofv3",
        "hip-bench",
    ):
        assert unexpected_option not in evaluate_help


def test_cli_help_documents_hip_cpp_compile_timeout_option():
    result = CliRunner().invoke(cli, ["evaluate", "--help"])
    assert result.exit_code == 0
    assert "--compile-timeout" in result.output
    assert "Compilation timeout in seconds" in result.output


def test_static_and_profile_docs_keep_diagnostic_only_authority_boundaries():
    static_docs = (REPO_ROOT / "docs/user/static_kernel_evidence.md").read_text()
    timing_docs = (REPO_ROOT / "docs/user/rocm_timing.md").read_text()
    claims_docs = (REPO_ROOT / "docs/user/CLAIMS.md").read_text()

    for docs in (static_docs, timing_docs):
        for expected in (
            "diagnostic",
            "not correctness authority",
            "performance authority",
            "timing authority",
            "score authority",
            "paper-parity authority",
            "leaderboard authority",
        ):
            assert expected in docs

    for expected in (
        "`rocprofv3` profiling as correctness or score authority",
        "Static Kernel Evidence as correctness authority",
        "Static Kernel Evidence as correctness, performance, timing, score",
    ):
        assert expected in claims_docs


def test_primary_cli_does_not_expose_v1_6_derived_workflow_options():
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    help_text = result.output

    for additive_non_primary_option in (
        "--amd-score-report",
        "--rocprofv3",
        "--timing-evidence",
        "--sol-bound",
        "--sol-bound-v2",
        "--bound-graph",
        "--extract-bound-graph",
        "--bound-estimates",
        "--formula-inputs",
        "--movement-bytes",
        "--operator-work-estimates",
        "--coverage-summary",
        "--aggregate-bound",
        "--hardware-model",
        "--amd-hardware-model",
        "--hardware-model-path",
    ):
        assert additive_non_primary_option not in help_text


def test_v1_10_solar_derivation_contract_keeps_claim_boundaries():
    contract = (REPO_ROOT / "docs/internal/solar_derivation_contract.md").read_text()

    for expected in (
        "sidecar-only",
        "not paper-scale dataset extraction",
        "not hosted leaderboard readiness",
        "not NVIDIA Blackwell/B200 equivalence",
        "not new real-hardware validation",
    ):
        assert expected in contract

    for fixture in load_solar_derivation_fixtures():
        boundary = json_dict(fixture["scope_boundary"])
        assert boundary["paper_scale_dataset"] is False, fixture["case_id"]
        assert boundary["hosted_leaderboard_ready"] is False, fixture["case_id"]
        assert boundary["nvidia_blackwell_b200_equivalence"] is False, fixture[
            "case_id"
        ]
        assert boundary["real_hardware_validation"] is False, fixture["case_id"]


def test_v1_10_solar_derivation_fields_remain_noncanonical():
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
    forbidden = (
        "solar_derivation",
        "semantic_groups",
        "semantic_axes",
        "source_kind",
        "source_detail",
        "confidence_rationale",
        "formula_provenance",
        "byte_provenance",
        "sol_execbench.solar_derivation.v1",
        "expected_subroles",
        "required_evidence",
        "missing_evidence",
        "warning_prefixes",
        "scope_boundary",
        "formula_evidence",
        "byte_evidence",
        "bound_evidence",
        "coverage_summary",
        "aggregate_status",
        "family_counts",
        "status_counts",
        "degraded_node_ids",
        "unsupported_node_ids",
        "estimated_node_ids",
        "score_eligible",
        "convolution_flops",
        "embedding_positional_bytes",
        "output_spatial",
        "selected_elements",
        "index_dtype",
        *PHASE50_INTERNAL_EVIDENCE_NAMES,
    )

    for payload in (
        definition.model_dump(mode="json"),
        workload.model_dump(mode="json"),
        trace.model_dump(mode="json"),
    ):
        for field in forbidden:
            assert field not in payload
            assert field not in repr(payload)

    canonical_trace_jsonl = json.dumps(trace.model_dump(mode="json"), sort_keys=True)
    for field in (
        *PHASE50_INTERNAL_EVIDENCE_NAMES,
        *PHASE51_INTERNAL_PUBLIC_BOUNDARY_FIELDS,
    ):
        assert field not in canonical_trace_jsonl


def test_amd_sol_artifacts_may_keep_existing_bound_fields_while_solar_fields_stay_sidecar_only():
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
    v3_payload = make_amd_sol_bound(
        definition,
        workload,
        hardware,
        hardware_model_ref="hardware/gfx1200.json",
    ).to_dict()

    assert "compute_bound_ms" in repr(v3_payload)
    for payload in (v3_payload,):
        assert "solar_derivation" not in payload
        payload_keys = _json_object_keys(payload)
        for field in PHASE51_SOLAR_ONLY_ARTIFACT_FIELDS:
            assert field not in payload_keys


def test_primary_cli_does_not_expose_v1_10_solar_derivation_options():
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    help_text = result.output

    for option in (
        "--solar-derivation",
        "--derive-solar",
        "--solar-fixtures",
        "--solar-contract",
        "--solar-sidecar",
        "--semantic-provenance",
        "--solar-evidence",
        "--solar-confidence",
        "--solar-provenance",
        "--derive-solar-sidecar",
        *PHASE50_INTERNAL_EVIDENCE_NAMES,
        *PHASE51_INTERNAL_PUBLIC_BOUNDARY_FIELDS,
    ):
        assert option not in help_text


def test_v1_11_dataset_manifest_keeps_acquisition_claim_boundary(tmp_path):
    problem_dir = tmp_path / "L1" / "demo"
    problem_dir.mkdir(parents=True)
    (problem_dir / "definition.json").write_text("{}\n", encoding="utf-8")
    (problem_dir / "workload.jsonl").write_text('{"uuid":"w"}\n', encoding="utf-8")

    manifest = build_dataset_manifest(
        tmp_path,
        categories=("L1",),
        source=DatasetManifestSource(revision="main"),
        created_at="2026-05-23T00:00:00Z",
    )
    boundary = manifest.claim_boundary

    assert boundary.acquisition_or_layout_complete is True
    assert boundary.rocm_readiness is False
    assert boundary.execution_success is False
    assert boundary.paper_level_validation is False
    assert boundary.hosted_leaderboard_parity is False
    assert boundary.upstream_solar_equivalence is False


def test_v1_11_dataset_docs_do_not_overclaim_acquisition_layout():
    docs = "\n".join(
        [
            (REPO_ROOT / "docs/user/GETTING-STARTED.md").read_text(),
            (REPO_ROOT / "docs/internal/analysis.md").read_text(),
        ]
    )

    assert "data/SOL-ExecBench/benchmark" in docs
    assert "acquisition/layout" in docs
    for expected_boundary in (
        "does not prove ROCm readiness",
        "execution success",
        "paper-level validation",
        "hosted leaderboard parity",
        "upstream SOLAR equivalence",
    ):
        assert expected_boundary in docs


def test_v1_11_inventory_readiness_fields_remain_sidecar_only():
    definition = make_definition(
        name="demo",
        axes={"N": {"type": "var"}},
        inputs={"x": {"shape": ["N"], "dtype": "float32"}},
        outputs={"out": {"shape": ["N"], "dtype": "float32"}},
        reference="def run(x):\n    return x",
    )
    workload = make_workload(axes={"N": 4}, inputs={"x": {"type": "random"}}, uuid="w")
    trace = make_trace(
        definition="demo", workload=workload, solution="solution", evaluation=None
    )
    forbidden = (
        "sol_execbench.dataset_inventory.v1",
        "sol_execbench.rocm_readiness.v1",
        "sol_execbench.ready_subset.v1",
        "readiness_checksum",
        "ready_subset_checksum",
        "ready_to_attempt_rocm_execution",
    )

    for payload in (
        definition.model_dump(mode="json"),
        workload.model_dump(mode="json"),
        trace.model_dump(mode="json"),
    ):
        text = json.dumps(payload, sort_keys=True)
        for field in forbidden:
            assert field not in text


def test_v1_11_execution_closure_fields_remain_sidecar_only():
    definition, workload, trace = _sample_definition_workload_trace()
    forbidden = (
        "sol_execbench.execution_closure.v1",
        "execution_closure",
        "execution_closure_checksum",
        "provenance_mismatches",
        "closure_status",
        "trace_ref",
        "source_refs",
        "skipped_existing_pass",
        "attempted_passed",
        "attempted_failed",
        "filtered",
        "readiness_blocked",
        "setup_blocked",
        "runtime_blocked",
        "missing_trace",
        "missing_derived_evidence",
        "derived_evidence_missing",
        "stale_provenance",
        "manifest_checksum_mismatch",
        "readiness_checksum_mismatch",
        "ready_subset_checksum_mismatch",
        "workload_identity_mismatch",
        "solution_mismatch",
        "solution_mode_mismatch",
        "evidence_requirement_mismatch",
        "bounded_ready_subset_execution",
        "full_235_problem_validation",
        "leaderboard_result",
        "score_authority",
    )

    for payload in (
        definition.model_dump(mode="json"),
        workload.model_dump(mode="json"),
        trace.model_dump(mode="json"),
    ):
        text = json.dumps(payload, sort_keys=True)
        for field in forbidden:
            assert field not in text


def test_v1_11_execution_closure_docs_keep_bounded_claim_boundary():
    docs = (REPO_ROOT / "docs/internal/analysis.md").read_text()

    assert "--execution-closure" in docs
    assert "bounded local execution audit" in docs
    assert "not full 235-problem validation" in docs
    assert "not paper parity" in docs
    assert "not a leaderboard result" in docs


def test_v1_19_paper_denominator_fields_remain_sidecar_only():
    definition, workload, trace = _sample_definition_workload_trace()
    forbidden = (
        "sol_execbench.paper_denominator_report.v1",
        "paper_denominator",
        "paper_denominator_checksum",
        "PaperDenominatorReport",
        "PaperDenominatorSourceRef",
        "source_refs",
        "amd_sol_artifacts",
        "solar_artifacts",
        "ready",
        "blocked",
        "unsupported",
        "deferred",
        "evidence_missing",
        "filtered",
        "skipped",
        "not_attempted",
        "paper_parity",
        "upstream_solar_parity",
        "leaderboard_authority",
        "native_host_validation",
        "new_hardware_validation",
        "full_235_problem_validation",
        "score_authority",
    )

    for payload in (
        definition.model_dump(mode="json"),
        workload.model_dump(mode="json"),
        trace.model_dump(mode="json"),
    ):
        text = json.dumps(payload, sort_keys=True)
        for field in forbidden:
            assert field not in text

    from sol_execbench.core.dataset.paper_denominator import (
        PaperDenominatorClaimBoundary,
    )

    sidecar_boundary = PaperDenominatorClaimBoundary().model_dump(mode="json")
    for field in (
        "paper_parity",
        "upstream_solar_parity",
        "leaderboard_authority",
        "native_host_validation",
        "new_hardware_validation",
        "full_235_problem_validation",
        "score_authority",
    ):
        assert sidecar_boundary[field] is False


def test_primary_cli_does_not_expose_v1_19_paper_denominator_options():
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    help_text = result.output

    for option in (
        "--paper-denominator",
        "--report-paper-denominator",
        "--paper-denominator-report",
        "--amd-sol-artifact",
        "--solar-artifact",
        "--created-at",
        "paper_denominator",
        "paper-denominator",
    ):
        assert option not in help_text


def test_v1_19_amd_bound_sanity_fields_remain_sidecar_only():
    definition, workload, trace = _sample_definition_workload_trace()
    forbidden = (
        "sol_execbench.amd_bound_sanity.v1",
        "amd_bound_sanity",
        "amd_bound_sanity_checksum",
        "AmdBoundSanityReport",
        "diagnostic_status",
        "diagnostic_flags",
        "status_totals",
        "provisional_rdna4_model_risk",
        "upstream_solar_equivalence",
        "amd_sol_model_validation",
        "solar_model_validation",
        "paper_parity",
        "leaderboard_authority",
        "score_authority_upgrade",
        "cdna3_validation",
        "mi300x_validation",
        "cdna4_validation",
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


def test_v1_19_amd_bound_sanity_does_not_enter_amd_score_contracts():
    from sol_execbench.core.scoring.amd_bound_sanity.models import (
        AMD_BOUND_SANITY_SCHEMA_VERSION,
        AmdBoundSanityClaimBoundary,
    )

    score_payload = build_amd_native_suite_report([]).to_dict()
    text = json.dumps(score_payload, sort_keys=True)

    for field in (
        AMD_BOUND_SANITY_SCHEMA_VERSION,
        "amd_bound_sanity",
        "diagnostic_status",
        "diagnostic_flags",
        "provisional_rdna4_model_risk",
        "score_authority_upgrade",
    ):
        assert field not in text

    boundary = AmdBoundSanityClaimBoundary().model_dump(mode="json")
    for field in (
        "upstream_solar_equivalence",
        "amd_sol_model_validation",
        "solar_model_validation",
        "paper_parity",
        "leaderboard_authority",
        "score_authority_upgrade",
        "cdna3_validation",
        "mi300x_validation",
        "cdna4_validation",
        "native_host_validation",
        "new_hardware_validation",
    ):
        assert boundary[field] is False

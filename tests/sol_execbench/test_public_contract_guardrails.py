from __future__ import annotations

import json
import sys
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
from sol_execbench.core.dataset import DatasetManifestSource, build_dataset_manifest
from sol_execbench.core.scoring.amd_score import (
    CDNA3_NO_VALIDATION_WARNING,
    DEGRADED_SOL_BOUND_WARNING,
    UNSCORED_SOL_BOUND_WARNING,
    build_amd_native_suite_report,
    score_amd_native_workload,
)
from sol_execbench.core.scoring.amd_sol import (
    EstimateConfidence,
    build_amd_sol_bound_artifact,
    default_amd_hardware_models,
)
from sol_execbench.core.scoring.amd_sol_v2 import build_amd_sol_bound_v2_artifact
from sol_execbench.core.scoring.amd_hardware_models import HardwareValidationStatus
from sol_execbench.core.data.definition import Definition
from sol_execbench.core.scoring import solar_derivation as solar_derivation_module
from sol_execbench.core.scoring.solar_derivation import SolarAggregateStatus
from sol_execbench_type_helpers import (
    json_dict,
    make_definition,
    make_solution,
    make_trace,
    make_workload,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PLANNING_ROOT = REPO_ROOT / ".planning"
COMPATIBILITY_INVENTORY = REPO_ROOT / "docs/internal/v1_4_compatibility_inventory.md"
TEST_DIR = str(Path(__file__).resolve().parent)
if TEST_DIR not in sys.path:
    sys.path.insert(0, TEST_DIR)

from solar_derivation_fixtures import load_solar_derivation_fixtures  # noqa: E402


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
    workload = make_workload(axes={"N": 16}, inputs={"x": {"type": "random"}}, uuid="w1")
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
        "static_kernel_evidence.v1",
        "sol_execbench.static_kernel_evidence.v1",
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


def test_cli_help_preserves_existing_public_options():
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    help_text = result.output
    for expected_option in (
        "Usage:",
        "--definition",
        "--workload",
        "--solution",
        "--config",
        "--compile-timeout",
        "--timeout",
        "--output",
        "--json",
        "--lock-clocks",
        "--keep-staging",
        "--profile",
        "--static-evidence",
        "--verbose",
    ):
        assert expected_option in help_text
    assert "contract" in help_text
    for unexpected_option in (
        "diagnose",
        "--rocprofv3",
        "hip-bench",
    ):
        assert unexpected_option not in help_text


def test_static_and_profile_docs_keep_diagnostic_only_authority_boundaries():
    static_docs = (REPO_ROOT / "docs/static_kernel_evidence.md").read_text()
    timing_docs = (REPO_ROOT / "docs/rocm_timing.md").read_text()
    claims_docs = (REPO_ROOT / "docs/CLAIMS.md").read_text()

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
        assert (
            boundary["nvidia_blackwell_b200_equivalence"] is False
        ), fixture["case_id"]
        assert boundary["real_hardware_validation"] is False, fixture["case_id"]


def test_v1_10_solar_derivation_fields_remain_noncanonical():
    definition = make_definition(
        name="demo",
        axes={"N": {"type": "var"}},
        inputs={"x": {"shape": ["N"], "dtype": "float32"}},
        outputs={"out": {"shape": ["N"], "dtype": "float32"}},
        reference="def run(x):\n    return x",
    )
    workload = make_workload(axes={"N": 16}, inputs={"x": {"type": "random"}}, uuid="w1")
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
    for field in (*PHASE50_INTERNAL_EVIDENCE_NAMES, *PHASE51_INTERNAL_PUBLIC_BOUNDARY_FIELDS):
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
    hardware = default_amd_hardware_models()["gfx1200"]
    v1_payload = build_amd_sol_bound_artifact(definition, workload, hardware).to_dict()
    v2_payload = build_amd_sol_bound_v2_artifact(
        definition,
        workload,
        hardware,
        hardware_model_ref="default_amd_hardware_models.gfx1200",
    ).to_dict()

    assert "compute_bound_ms" in repr(v1_payload)
    assert "compute_bound_ms" in repr(v2_payload)
    for payload in (v1_payload, v2_payload):
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
            (REPO_ROOT / "docs/GETTING-STARTED.md").read_text(),
            (REPO_ROOT / "docs/analysis.md").read_text(),
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
    trace = make_trace(definition="demo", workload=workload, solution="solution", evaluation=None)
    forbidden = (
        "sol_execbench.dataset_inventory.v1",
        "sol_execbench.rocm_readiness.v1",
        "sol_execbench.ready_subset.v1",
        "readiness_checksum",
        "ready_subset_checksum",
        "ready_to_attempt_rocm_execution",
    )

    for payload in (definition.model_dump(mode="json"), workload.model_dump(mode="json"), trace.model_dump(mode="json")):
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
    docs = (REPO_ROOT / "docs/analysis.md").read_text()

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

    from sol_execbench.core.dataset.paper_denominator import PaperDenominatorClaimBoundary

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
    from sol_execbench.core.scoring.amd_bound_sanity import (
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
    from sol_execbench.core.scoring.amd_bound_sanity import (
        build_amd_bound_sanity_report,
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
    docs = (REPO_ROOT / "docs/analysis.md").read_text()

    assert "scripts/report_parity_gaps.py" in docs
    assert "discovered, parsed, ready, blocked" in docs
    assert "not full validation" in docs
    assert "not paper parity" in docs
    assert "not upstream SOLAR parity" in docs
    assert "not NVIDIA B200 or" in docs
    assert "not hosted leaderboard readiness" in docs


def test_v1_11_release_closure_summarizes_remaining_claim_gaps():
    doc = (REPO_ROOT / "docs/v1_11_release_closure.md").read_text()

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
    artifact = build_amd_sol_bound_artifact(
        definition,
        workload,
        default_amd_hardware_models()["gfx1200"],
    )

    payload = score_amd_native_workload(
        artifact,
        measured_latency_ms=1.0,
        baseline_latency_ms=2.0,
        trace_ref="traces/matmul.json",
        timing_evidence_ref="timing/matmul.json",
        sol_bound_ref="bounds/matmul.json",
        baseline_ref="baseline/matmul.json",
        hardware_model_ref="default_amd_hardware_models.gfx1200",
        derived_evidence_refs={
            "formula": "solar/matmul.json#groups.formula_evidence",
            "hardware_model": "default_amd_hardware_models.gfx1200",
            "coverage": "solar/matmul.json#coverage_summary",
            "score_eligibility": "solar/matmul.json#aggregate_status",
        },
    ).to_dict()

    assert set(payload["evidence_refs"]) == PUBLIC_SCORE_EVIDENCE_REF_KEYS
    assert set(payload["derived_evidence_refs"]) == DERIVED_REPORT_EVIDENCE_REF_KEYS
    assert set(payload["evidence_refs"]).isdisjoint(
        {"formula", "coverage", "score_eligibility"}
    )


def test_degraded_complex_family_score_eligibility_ignores_solar_sidecars():
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
    hardware = default_amd_hardware_models()["gfx1200"]

    for definition, workload, formula_kind in cases:
        artifact = build_amd_sol_bound_v2_artifact(definition, workload, hardware)
        score = score_amd_native_workload(
            artifact,
            measured_latency_ms=1.0,
            baseline_latency_ms=2.0,
            hardware_model_ref="default_amd_hardware_models.gfx1200",
        )

        assert artifact.aggregate_bound.status == "degraded"
        assert any(
            estimate["formula_kind"] == formula_kind
            for estimate in artifact.operator_work_estimates
        )
        assert score.supported is True
        assert score.claim_level == "amd-native-derived"
        assert DEGRADED_SOL_BOUND_WARNING in score.warnings
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
    hardware = default_amd_hardware_models()["gfx1200"]
    v1_artifact = build_amd_sol_bound_artifact(definition, workload, hardware)
    v2_artifact = build_amd_sol_bound_v2_artifact(
        definition,
        workload,
        hardware,
        hardware_model_ref="default_amd_hardware_models.gfx1200",
    )

    v1_score = score_amd_native_workload(
        v1_artifact,
        measured_latency_ms=1.0,
        baseline_latency_ms=2.0,
        hardware_model_ref="default_amd_hardware_models.gfx1200",
    )
    v2_score = score_amd_native_workload(
        v2_artifact,
        measured_latency_ms=1.0,
        baseline_latency_ms=2.0,
        hardware_model_ref="default_amd_hardware_models.gfx1200",
    )

    assert v1_score.supported is True
    assert v2_score.supported is True
    assert v1_score.to_dict()["claim_level"] == "amd-native-derived"
    assert v2_score.to_dict()["claim_level"] == "amd-native-derived"
    for field in PHASE51_SCORE_INTERNAL_EVIDENCE_REFS:
        assert field not in v1_score.to_dict()["evidence_refs"]
        assert field not in v2_score.to_dict()["evidence_refs"]
    v1_artifact_keys = _json_object_keys(v1_artifact.to_dict())
    v2_artifact_keys = _json_object_keys(v2_artifact.to_dict())
    for field in PHASE51_SOLAR_ONLY_ARTIFACT_FIELDS:
        assert field not in v1_artifact_keys
        assert field not in v2_artifact_keys


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
    artifact = build_amd_sol_bound_artifact(
        definition,
        workload,
        default_amd_hardware_models()["gfx1200"],
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
    if not (PLANNING_ROOT / "PROJECT.md").exists() or not (
        PLANNING_ROOT / "REQUIREMENTS.md"
    ).exists():
        pytest.skip("SOL planning metadata is not present in this nested checkout")
    project = (PLANNING_ROOT / "PROJECT.md").read_text()
    requirements = (PLANNING_ROOT / "REQUIREMENTS.md").read_text()
    analysis = Path("docs/analysis.md").read_text()

    assert "CDNA 3 (`gfx94*`) full adapted suite validation remains deferred" in project
    assert "CDNA3-family, CDNA4, or native-host ROCm validation expansion" in requirements
    assert "MI300X is the CDNA3 hardware target rather than a separate architecture target" in requirements
    assert "not NVIDIA B200, SOLAR, or leaderboard equivalence claims" in analysis
    assert "hardware_validation_status" in analysis
    assert "model_validation_status" in analysis


def test_hardware_model_evidence_survives_bound_and_score_artifacts():
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
    hardware = default_amd_hardware_models()["gfx1200"]
    artifact = build_amd_sol_bound_artifact(definition, workload, hardware)
    score = score_amd_native_workload(
        artifact,
        measured_latency_ms=1.0,
        baseline_latency_ms=2.0,
        hardware_model_ref="default_amd_hardware_models.gfx1200",
    )
    payload = artifact.to_dict()

    assert payload["hardware_model"]["source"]
    assert payload["hardware_model"]["confidence"] == EstimateConfidence.INEXACT.value
    assert (
        payload["hardware_model"]["model_validation_status"]
        == HardwareValidationStatus.PROVISIONAL.value
    )
    assert "validation_status" not in payload["hardware_model"]
    assert score.evidence_refs["hardware_model"] == "default_amd_hardware_models.gfx1200"


def test_definition_workload_trace_schemas_do_not_include_derived_artifact_fields():
    definition = make_definition(
        name="demo",
        axes={"N": {"type": "var"}},
        inputs={"x": {"shape": ["N"], "dtype": "float32"}},
        outputs={"out": {"shape": ["N"], "dtype": "float32"}},
        reference="def run(x):\n    return x",
    )
    workload = make_workload(axes={"N": 16}, inputs={"x": {"type": "random"}}, uuid="w1")
    trace = make_trace(
        definition="demo",
        workload=workload,
        solution="solution",
        evaluation=None,
    )

    assert "hardware_model" not in definition.model_dump(mode="json")
    assert "bound_graph" not in definition.model_dump(mode="json")
    assert "graph_nodes" not in definition.model_dump(mode="json")
    assert "op_family" not in definition.model_dump(mode="json")
    assert "op_bounds" not in definition.model_dump(mode="json")
    assert "formula_kind" not in definition.model_dump(mode="json")
    assert "read_bytes" not in definition.model_dump(mode="json")
    assert "movement_bytes" not in definition.model_dump(mode="json")
    assert "operator_work_estimates" not in definition.model_dump(mode="json")
    for field in PHASE51_INTERNAL_PUBLIC_BOUNDARY_FIELDS:
        assert field not in definition.model_dump(mode="json")
        assert field not in workload.model_dump(mode="json")
        assert field not in trace.model_dump(mode="json")
    assert "aggregate_bound" not in definition.model_dump(mode="json")
    assert "hardware_model_ref" not in definition.model_dump(mode="json")
    assert "hardware_model" not in workload.model_dump(mode="json")
    assert "bound_graph" not in workload.model_dump(mode="json")
    assert "op_family" not in workload.model_dump(mode="json")
    assert "formula_kind" not in workload.model_dump(mode="json")
    assert "read_bytes" not in workload.model_dump(mode="json")
    assert "movement_bytes" not in workload.model_dump(mode="json")
    assert "operator_work_estimates" not in workload.model_dump(mode="json")
    assert "aggregate_bound" not in workload.model_dump(mode="json")
    assert "hardware_model_ref" not in workload.model_dump(mode="json")
    assert "bound_graph" not in trace.model_dump(mode="json")
    assert "graph_nodes" not in trace.model_dump(mode="json")
    assert "op_family" not in trace.model_dump(mode="json")
    assert "amd_native" not in trace.model_dump(mode="json")
    assert "formula_kind" not in trace.model_dump(mode="json")
    assert "read_bytes" not in trace.model_dump(mode="json")
    assert "movement_bytes" not in trace.model_dump(mode="json")
    assert "operator_work_estimates" not in trace.model_dump(mode="json")
    assert "aggregate_bound" not in trace.model_dump(mode="json")
    assert "hardware_model_ref" not in trace.model_dump(mode="json")


def test_v1_4_compatibility_inventory_covers_public_contracts():
    text = COMPATIBILITY_INVENTORY.read_text()
    for heading in (
        "Public CLI Contract",
        "Definition Schema Contract",
        "Workload Schema Contract",
        "Solution Format Contract",
        "Trace JSONL Contract",
        "Eval-Driver Semantics Contract",
        "Phase 19 Non-Goals",
    ):
        assert heading in text
    for source_ref in (
        "src/sol_execbench/cli/main.py",
        "src/sol_execbench/core/data/definition.py",
        "src/sol_execbench/core/data/workload.py",
        "src/sol_execbench/core/data/solution.py",
        "src/sol_execbench/core/data/trace.py",
        "src/sol_execbench/driver/templates/eval_driver.py",
    ):
        assert source_ref in text


def test_v1_4_compatibility_inventory_rejects_phase_19_public_drift():
    text = COMPATIBILITY_INVENTORY.read_text()
    for invariant in (
        "Do not add public `sol-execbench` CLI options or subcommands.",
        "Do not change Pydantic public field names",
        "Do not add fields to trace JSONL.",
        "Do not replace the eval driver",
        "Do not claim CDNA 3 hardware validation.",
        "Do not introduce the hip-execbench TypeScript/Zod runtime stack.",
    ):
        assert invariant in text


def test_public_example_paths_remain_hip_facing():
    example_root = Path("examples/hip_cpp")
    solution_files = sorted(example_root.glob("*/solution_hip.json"))
    assert solution_files, "expected HIP-facing public native examples"
    assert not list(example_root.glob("*/solution_cuda.json"))


def test_cdna3_validation_remains_deferred_in_docs():
    handoff = Path(".planning/milestones/CDNA3-VALIDATION-HANDOFF.md").read_text()
    project = Path(".planning/PROJECT.md").read_text()
    current_requirements = Path(".planning/REQUIREMENTS.md")
    requirements = (
        current_requirements
        if current_requirements.exists()
        else Path(".planning/milestones/v1.28-REQUIREMENTS.md")
    ).read_text()
    roadmap = Path(".planning/ROADMAP.md").read_text()
    assert "Status:** Deferred to next milestone" in handoff
    assert "Actual CDNA3/MI300X full-suite execution" in project
    assert "current machine cannot" in project
    assert "actual full-suite" in project
    assert "Actual CDNA3/MI300X full-suite execution in v1.28" in requirements
    assert "hardware validation remains deferred" in roadmap
    assert "requires_cdna3" in roadmap
    assert "CDNA3 full-suite validation has not been recorded" in CDNA3_NO_VALIDATION_WARNING
    assert "hardware-validation claim" in CDNA3_NO_VALIDATION_WARNING

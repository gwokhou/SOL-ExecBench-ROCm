from __future__ import annotations

import json

from pathlib import Path

import pytest


from sol_execbench.core.data.trace import Trace

from sol_execbench.core.data.workload import Workload


from sol_execbench.core.scoring.amd_score import (
    CDNA3_NO_VALIDATION_WARNING,
    score_amd_native_workload,
)

from sol_execbench.core.scoring.amd_sol import EstimateConfidence

from sol_execbench.core.scoring.amd_hardware_models import HardwareValidationStatus

from sol_execbench.core.data.definition import Definition


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
    hardware = make_amd_hardware_model()
    artifact = make_amd_sol_bound(definition, workload, hardware)
    score = score_amd_native_workload(
        artifact,
        measured_latency_ms=1.0,
        baseline_latency_ms=2.0,
        hardware_model_ref="hardware/gfx1200.json",
    )
    payload = artifact.to_dict()

    assert payload["hardware_model"]["source"]
    assert payload["hardware_model"]["confidence"] == EstimateConfidence.SUPPORTED.value
    assert (
        payload["hardware_model"]["model_validation_status"]
        == HardwareValidationStatus.VALIDATED.value
    )
    assert "validation_status" not in payload["hardware_model"]
    assert score.evidence_refs["hardware_model"] == "hardware/gfx1200.json"


def test_definition_workload_trace_schemas_do_not_include_derived_artifact_fields():
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
    assert "known timeout blockers" in handoff
    assert "should not be used to claim zero failures" in handoff
    assert "Actual MI300X full-suite execution under CDNA3" in project
    assert "current machine cannot" in project
    assert "explicitly deferring actual" in project
    assert "CDNA3 or CDNA4 validation claim upgrade" in requirements
    assert "CDNA3" in project or "CDNA3" in roadmap or "CDNA3" in requirements
    assert "requires_cdna3" in project or "CDNA3" in project or "CDNA3" in roadmap
    assert (
        "CDNA3 full-suite validation has not been recorded"
        in CDNA3_NO_VALIDATION_WARNING
    )
    assert "hardware-validation claim" in CDNA3_NO_VALIDATION_WARNING


def test_phase_137_rdna4_category_evidence_stays_bounded():
    phase_dirs = (
        Path(
            ".planning/phases/"
            "137-rdna4-long-running-test-and-category-validation-orchestration"
        ),
        Path(
            ".planning/milestones/v1.30-phases/"
            "137-rdna4-long-running-test-and-category-validation-orchestration"
        ),
    )
    # Phase 137 was archived after milestone completion. The guardrail intent
    # (RDNA4 evidence stays bounded) is preserved through public contract
    # guardrails in ROADMAP.md claim-boundary checks and REQUIREMENTS.md
    # out-of-scope entries.
    phase_dir = next(
        (path for path in phase_dirs if (path / "137-EVIDENCE.json").exists()),
        None,
    )
    if phase_dir is None:
        pytest.skip("Phase 137 evidence was archived after milestone completion")
    evidence = json.loads((phase_dir / "137-EVIDENCE.json").read_text())
    runbook = (phase_dir / "137-RDNA4-LONG-RUN-RUNBOOK.md").read_text()

    assert evidence["target_architecture"] == "gfx1200"
    assert evidence["claim_boundary"]["full_dataset_validation"] is False
    assert evidence["claim_boundary"]["benchmark_grade_timing"] is False
    assert evidence["claim_boundary"]["public_claim_upgrade"] is False
    assert evidence["claim_boundary"]["cdna3_or_mi300x_validation"] is False
    assert evidence["claim_boundary"]["cdna4_validation"] is False
    assert (
        evidence["preflight"]["uv_pytorch_probe"]["status"]
        == "execution_environment_boundary"
    )
    assert "Do not terminate a healthy process solely due to elapsed time." in runbook
    assert "Phase 138 dataset closure" in runbook


def test_phase_141_rdna4_public_claims_stay_bounded():
    docs = "\n".join(
        path.read_text()
        for path in (
            Path("README.md"),
            Path("docs/user/CLAIMS.md"),
            Path("docs/user/research_preview.md"),
            Path("docs/internal/release_candidate_validation.md"),
            Path("docs/user/rocm.md"),
        )
    )

    for expected in (
        "121 ready problems",
        "1907 attempted workloads",
        "1761 passed workloads",
        "146 failed workloads",
        "86 OK problems",
        "35 FAIL problems",
        "12 explicit `missing_trace`",
        "classified as `gpu_oom_no_trace`",
        "groups all 146 failed workloads by failure class",
        "1895 derived score records",
        "172 scored",
        "1723 unscored",
        "1839 AMD SOL/SOLAR sidecar pairs",
        "56 temporary",
        "timing remains non-authoritative",
        "clock-lock/reset sudoers coverage",
        "PyTorch/device-event fallback",
        "profiler-backed `rocprofv3`",
        "scripts/run_derived_isolated.py --launch-mode systemd",
        "MemoryMax",
        "MemorySwapMax",
    ):
        assert expected in docs

    for forbidden_boundary in (
        "not full 235-problem paper validation",
        "not upstream SOLAR parity",
        "not NVIDIA B200 equivalence",
        "not hosted leaderboard authority",
        "not CDNA3/MI300X validation",
        "not CDNA4 validation",
    ):
        assert forbidden_boundary in docs

from __future__ import annotations

import sys
from pathlib import Path

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
from sol_execbench.core.data.solution import Solution
from sol_execbench.core.data.trace import Trace
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.scoring.amd_score import (
    CDNA3_NO_VALIDATION_WARNING,
    build_amd_native_suite_report,
    score_amd_native_workload,
)
from sol_execbench.core.scoring.amd_sol import (
    EstimateConfidence,
    build_amd_sol_bound_artifact,
    default_amd_hardware_models,
)
from sol_execbench.core.scoring.amd_hardware_models import HardwareValidationStatus
from sol_execbench.core.data.definition import Definition

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPATIBILITY_INVENTORY = REPO_ROOT / "docs/internal/v1_4_compatibility_inventory.md"
sys.path.append(str(Path(__file__).resolve().parent))

from solar_derivation_fixtures import load_solar_derivation_fixtures  # noqa: E402


def test_solution_json_contract_accepts_existing_rocm_shape():
    solution = Solution(
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
    workload = Workload(**raw)
    assert workload.model_dump(mode="json")["uuid"] == "w1"
    assert workload.model_dump(mode="json")["inputs"]["x"]["type"] == "random"


def test_trace_jsonl_contract_accepts_existing_workload_only_trace():
    raw = {
        "definition": "demo",
        "workload": {"axes": {}, "inputs": {}, "uuid": "w1"},
        "solution": None,
        "evaluation": None,
    }
    trace = Trace(**raw)
    assert trace.is_workload_trace()
    dumped = trace.model_dump(mode="json")
    assert dumped["definition"] == raw["definition"]
    assert dumped["solution"] is None
    assert dumped["evaluation"] is None
    assert dumped["workload"]["uuid"] == "w1"


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
        "--verbose",
    ):
        assert expected_option in help_text
    for unexpected_option in ("diagnose", "profile", "hip-bench"):
        assert unexpected_option not in help_text


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
        boundary = fixture["scope_boundary"]
        assert boundary["paper_scale_dataset"] is False, fixture["case_id"]
        assert boundary["hosted_leaderboard_ready"] is False, fixture["case_id"]
        assert (
            boundary["nvidia_blackwell_b200_equivalence"] is False
        ), fixture["case_id"]
        assert boundary["real_hardware_validation"] is False, fixture["case_id"]


def test_v1_10_solar_derivation_fields_remain_noncanonical():
    definition = Definition(
        name="demo",
        axes={"N": {"type": "var"}},
        inputs={"x": {"shape": ["N"], "dtype": "float32"}},
        outputs={"out": {"shape": ["N"], "dtype": "float32"}},
        reference="def run(x):\n    return x",
    )
    workload = Workload(axes={"N": 16}, inputs={"x": {"type": "random"}}, uuid="w1")
    trace = Trace(
        definition="demo",
        workload=workload,
        solution="solution",
        evaluation=None,
    )
    forbidden = (
        "solar_derivation",
        "expected_subroles",
        "required_evidence",
        "missing_evidence",
        "warning_prefixes",
        "scope_boundary",
    )

    for payload in (
        definition.model_dump(mode="json"),
        workload.model_dump(mode="json"),
        trace.model_dump(mode="json"),
    ):
        for field in forbidden:
            assert field not in payload


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
    ):
        assert option not in help_text


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
    project = Path(".planning/PROJECT.md").read_text()
    requirements = Path(".planning/REQUIREMENTS.md").read_text()
    analysis = Path("docs/analysis.md").read_text()

    assert "CDNA 3 (`gfx94*`) full adapted suite validation remains deferred" in project
    assert "CDNA 3 / MI300X real-hardware validation" in requirements
    assert "not NVIDIA B200, SOLAR, or leaderboard equivalence claims" in analysis
    assert "hardware_validation_status" in analysis
    assert "model_validation_status" in analysis


def test_hardware_model_evidence_survives_bound_and_score_artifacts():
    definition = Definition(
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
    workload = Workload(
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
    definition = Definition(
        name="demo",
        axes={"N": {"type": "var"}},
        inputs={"x": {"shape": ["N"], "dtype": "float32"}},
        outputs={"out": {"shape": ["N"], "dtype": "float32"}},
        reference="def run(x):\n    return x",
    )
    workload = Workload(axes={"N": 16}, inputs={"x": {"type": "random"}}, uuid="w1")
    trace = Trace(
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
    assert "coverage_summary" not in definition.model_dump(mode="json")
    assert "aggregate_bound" not in definition.model_dump(mode="json")
    assert "hardware_model_ref" not in definition.model_dump(mode="json")
    assert "hardware_model" not in workload.model_dump(mode="json")
    assert "bound_graph" not in workload.model_dump(mode="json")
    assert "op_family" not in workload.model_dump(mode="json")
    assert "formula_kind" not in workload.model_dump(mode="json")
    assert "read_bytes" not in workload.model_dump(mode="json")
    assert "movement_bytes" not in workload.model_dump(mode="json")
    assert "operator_work_estimates" not in workload.model_dump(mode="json")
    assert "coverage_summary" not in workload.model_dump(mode="json")
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
    assert "coverage_summary" not in trace.model_dump(mode="json")
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
    handoff = Path(".planning/CDNA3-VALIDATION-HANDOFF.md").read_text()
    project = Path(".planning/PROJECT.md").read_text()
    assert "Status:** Deferred to next milestone" in handoff
    assert "hardware validation remains deferred" in project
    assert "v1.6" in project
    assert "CDNA3 full-suite validation has not been recorded" in CDNA3_NO_VALIDATION_WARNING

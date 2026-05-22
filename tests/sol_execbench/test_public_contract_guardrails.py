from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from sol_execbench.cli.main import cli
from sol_execbench.core.data.solution import Solution
from sol_execbench.core.data.trace import Trace
from sol_execbench.core.data.workload import Workload

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPATIBILITY_INVENTORY = REPO_ROOT / "docs/internal/v1_4_compatibility_inventory.md"


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

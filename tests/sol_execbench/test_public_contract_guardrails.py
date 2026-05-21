from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from sol_execbench.cli.main import cli
from sol_execbench.core.data.solution import Solution
from sol_execbench.core.data.trace import Trace
from sol_execbench.core.data.workload import Workload


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
    assert "Usage:" in help_text
    assert "--definition" in help_text
    assert "--workload" in help_text
    assert "--solution" in help_text
    assert "--json" in help_text
    assert "diagnose" not in help_text


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

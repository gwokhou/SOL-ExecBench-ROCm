from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from sol_execbench_type_helpers import make_definition, make_workload


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_PATH = REPO_ROOT / "scripts" / "internal" / "audit_ast_operator_coverage.py"
spec = importlib.util.spec_from_file_location(
    "audit_ast_operator_coverage", SCRIPT_PATH
)
assert spec is not None
audit_ast_operator_coverage = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = audit_ast_operator_coverage
spec.loader.exec_module(audit_ast_operator_coverage)


def _write_problem(root: Path, reference: str) -> None:
    problem = root / "L1" / "demo"
    problem.mkdir(parents=True)
    definition = make_definition(
        name="ast_coverage_demo",
        axes={"N": {"type": "var"}},
        inputs={"x": {"shape": ["N"], "dtype": "float32"}},
        outputs={"out": {"shape": ["N"], "dtype": "float32"}},
        reference=reference,
    )
    workload = make_workload(
        axes={"N": 64}, inputs={"x": {"type": "random"}}, uuid="w1"
    )
    (problem / "definition.json").write_text(
        definition.model_dump_json(), encoding="utf-8"
    )
    (problem / "workload.jsonl").write_text(
        workload.model_dump_json() + "\n", encoding="utf-8"
    )


def test_ast_operator_coverage_audit_accepts_inexact_but_covered_graph(tmp_path):
    _write_problem(
        tmp_path,
        "def run(x):\n"
        "    def helper(value):\n"
        "        return value.sin()\n"
        "    for i in range(x.size(0)):\n"
        "        x[i] = helper(x[i])\n"
        "    return x\n",
    )

    result = audit_ast_operator_coverage.audit(tmp_path)

    assert result["problem_count"] == 1
    assert result["workload_count"] == 1
    assert result["gap_count"] == 0
    assert result["error_count"] == 0


def test_ast_operator_coverage_audit_reports_unknown_operator(tmp_path):
    _write_problem(tmp_path, "def run(x):\n    return unknown_tensor_op(x)\n")

    result = audit_ast_operator_coverage.audit(tmp_path)

    assert result["gap_count"] == 1
    assert result["gaps"][0]["operators"] == [
        "unknown_tensor_op",
        "unsupported_operator:unknown_tensor_op",
    ]


def test_ast_operator_coverage_audit_checks_definition_without_workloads(tmp_path):
    definitions = tmp_path / "trace" / "definitions" / "sampling"
    definitions.mkdir(parents=True)
    definition = make_definition(
        name="sampling_demo",
        axes={"N": {"type": "var"}},
        inputs={"x": {"shape": ["N"], "dtype": "float32"}},
        outputs={"out": {"shape": [], "dtype": "int64"}},
        reference="import torch\ndef run(x):\n    return torch.multinomial(x, 1)\n",
    )
    (definitions / "sampling.json").write_text(
        json.dumps(definition.model_dump(mode="json")), encoding="utf-8"
    )
    (tmp_path / "trace" / "workloads").mkdir()

    result = audit_ast_operator_coverage.audit(tmp_path)

    assert result["syntax_only_problem_count"] == 1
    assert result["gap_count"] == 0

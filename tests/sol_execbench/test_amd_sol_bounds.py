from __future__ import annotations

import json
from pathlib import Path

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.trace import (
    Correctness,
    Environment,
    Evaluation,
    EvaluationStatus,
    Performance,
    Trace,
)
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.scoring.amd_sol import (
    AMD_SOL_SCHEMA_VERSION,
    EstimateConfidence,
    HardwareValidationStatus,
    WorkEstimate,
    build_amd_sol_bound_artifact,
    default_amd_hardware_models,
    estimate_work,
    extract_graph,
    summarize_amd_sol_coverage,
)
from sol_execbench.core.scoring.amd_bound_estimates import estimate_bound_work
from sol_execbench.core.scoring.amd_bound_graph import build_bound_graph
from sol_execbench.core.scoring.amd_hardware_models import load_amd_hardware_model


REPO_ROOT = Path(__file__).resolve().parents[2]


def _matmul_definition() -> Definition:
    return Definition(
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


def test_matmul_bound_artifact_records_graph_work_hardware_and_bounds():
    definition = _matmul_definition()
    workload = Workload(
        axes={"M": 2},
        inputs={"a": {"type": "random"}, "b": {"type": "random"}},
        uuid="matmul-workload",
    )
    hardware = default_amd_hardware_models()["gfx1200"]

    artifact = build_amd_sol_bound_artifact(definition, workload, hardware)
    payload = artifact.to_dict()

    assert artifact.schema_version == AMD_SOL_SCHEMA_VERSION
    assert artifact.derived is True
    assert artifact.graph_nodes[0].op_type == "matmul"
    assert artifact.work_estimates[0].confidence == EstimateConfidence.SUPPORTED
    assert artifact.work_estimates[0].flops == 128.0
    assert artifact.work_estimates[0].bytes_accessed == 224.0
    assert artifact.op_bounds[0].sol_bound_ms > 0.0
    assert artifact.aggregate_sol_bound_ms == artifact.op_bounds[0].sol_bound_ms
    assert payload["coverage_summary"]["supported_ops"] == 1
    assert payload["hardware_model"]["architecture"] == "gfx1200"
    assert payload["hardware_model"]["hardware_validation_status"] == "validated"
    assert payload["hardware_model"]["model_validation_status"] == "provisional"
    assert "formula_inputs" not in json.dumps(payload)
    assert "read_bytes" not in json.dumps(payload)
    assert "movement_bytes" not in json.dumps(payload)
    assert "operator_work_estimates" not in json.dumps(payload)


def test_elementwise_work_estimate_is_inexact_and_auditable():
    definition = Definition(
        name="add_demo",
        axes={"N": {"type": "var"}},
        inputs={
            "x": {"shape": ["N"], "dtype": "float32"},
            "y": {"shape": ["N"], "dtype": "float32"},
        },
        outputs={"out": {"shape": ["N"], "dtype": "float32"}},
        reference="def run(x, y):\n    return x + y",
    )
    workload = Workload(
        axes={"N": 16},
        inputs={"x": {"type": "random"}, "y": {"type": "random"}},
        uuid="add-workload",
    )

    graph = extract_graph(definition)
    estimates = estimate_work(definition, workload, graph)
    rich_estimates = estimate_bound_work(build_bound_graph(definition, workload))

    assert graph[0].op_type == "elementwise"
    assert estimates[0].confidence == EstimateConfidence.INEXACT
    assert estimates[0].flops == 16.0
    assert estimates[0].bytes_accessed == 192.0
    assert estimates[0].bytes_accessed == rich_estimates[0].total_bytes
    assert "one operation per output element" in estimates[0].rationale


def test_unsupported_ops_stay_visible_instead_of_getting_silent_scores(tmp_path: Path):
    definition = Definition(
        name="unsupported_demo",
        axes={"N": {"type": "var"}},
        inputs={"x": {"shape": ["N", "N"], "dtype": "float32"}},
        outputs={"out": {"shape": ["N", "N"], "dtype": "float32"}},
        reference="import torch\n\ndef run(x):\n    return torch.linalg.inv(x)",
    )
    workload = Workload(
        axes={"N": 4},
        inputs={"x": {"type": "random"}},
        uuid="unsupported-workload",
    )
    hardware = _cdna3_model(tmp_path)

    artifact = build_amd_sol_bound_artifact(definition, workload, hardware)

    assert artifact.graph_nodes[0].op_type == "unsupported"
    assert artifact.work_estimates[0].confidence == EstimateConfidence.UNSUPPORTED
    assert artifact.work_estimates[0].bytes_accessed == 0.0
    assert artifact.op_bounds[0].confidence == EstimateConfidence.UNSUPPORTED
    assert artifact.hardware_model.model_validation_status == HardwareValidationStatus.UNVALIDATED
    assert artifact.hardware_model.hardware_validation_status == HardwareValidationStatus.UNVALIDATED
    assert artifact.hardware_model.source.endswith("CDNA3 scaffold for phase 45")


def test_legacy_work_estimate_fields_are_unchanged_and_adapt_rich_totals():
    definition = _matmul_definition()
    workload = Workload(
        axes={"M": 2},
        inputs={"a": {"type": "random"}, "b": {"type": "random"}},
        uuid="matmul-workload",
    )
    graph = extract_graph(definition)

    estimates = estimate_work(definition, workload, graph)
    rich_estimates = estimate_bound_work(build_bound_graph(definition, workload))

    assert tuple(WorkEstimate.__dataclass_fields__) == (
        "node_id",
        "flops",
        "bytes_accessed",
        "confidence",
        "rationale",
    )
    assert estimates[0].node_id == rich_estimates[0].node_id
    assert estimates[0].flops == 128.0
    assert estimates[0].bytes_accessed == rich_estimates[0].total_bytes


def test_coverage_summary_counts_supported_inexact_and_unsupported_ops():
    definition = Definition(
        name="coverage_demo",
        axes={"N": {"type": "var"}},
        inputs={"x": {"shape": ["N"], "dtype": "float32"}},
        outputs={"out": {"shape": ["N"], "dtype": "float32"}},
        reference=(
            "import torch\n\n"
            "def run(x):\n"
            "    y = torch.relu(x)\n"
            "    z = y.sum()\n"
            "    return z.reshape(1)\n"
        ),
    )
    workload = Workload(
        axes={"N": 8},
        inputs={"x": {"type": "random"}},
        uuid="coverage-workload",
    )

    graph = extract_graph(definition)
    estimates = estimate_work(definition, workload, graph)
    summary = summarize_amd_sol_coverage(graph, estimates)
    payload = summary.to_dict()

    assert summary.derived is True
    assert summary.total_ops == 3
    assert summary.supported_ops == 0
    assert summary.inexact_ops == 3
    assert summary.unsupported_ops == 0
    assert payload["op_type_counts"] == {
        "activation": 1,
        "reduction": 1,
        "data_movement": 1,
    }


def test_reduction_data_movement_and_softmax_estimates_are_labeled():
    definition = Definition(
        name="softmax_demo",
        axes={"N": {"type": "var"}, "M": {"type": "const", "value": 4}},
        inputs={"x": {"shape": ["N", "M"], "dtype": "float32"}},
        outputs={"out": {"shape": ["N", "M"], "dtype": "float32"}},
        reference=(
            "import torch\n\n"
            "def run(x):\n"
            "    y = x.transpose(0, 1)\n"
            "    z = torch.softmax(y, dim=-1)\n"
            "    return z.reshape(x.shape)\n"
        ),
    )
    workload = Workload(
        axes={"N": 8},
        inputs={"x": {"type": "random"}},
        uuid="softmax-workload",
    )

    graph = extract_graph(definition)
    estimates = estimate_work(definition, workload, graph)

    assert [node.op_type for node in graph] == [
        "data_movement",
        "softmax",
        "data_movement",
    ]
    assert all(estimate.confidence == EstimateConfidence.INEXACT for estimate in estimates)
    assert estimates[0].flops == 0.0
    assert estimates[1].flops > 0.0
    assert "softmax-like" in estimates[1].rationale


def test_amd_sol_artifacts_do_not_mutate_canonical_trace_payloads():
    trace = Trace(
        definition="matmul_demo",
        workload=Workload(
            axes={"M": 2},
            inputs={"a": {"type": "random"}, "b": {"type": "random"}},
            uuid="matmul-workload",
        ),
        solution="solution",
        evaluation=Evaluation(
            status=EvaluationStatus.PASSED,
            environment=Environment(hardware="AMD gfx1200", libs={}),
            timestamp="2026-05-22T00:00:00Z",
            correctness=Correctness(),
            performance=Performance(
                latency_ms=1.0,
                reference_latency_ms=2.0,
                speedup_factor=2.0,
            ),
        ),
    )
    before = trace.model_dump(mode="json")

    artifact = build_amd_sol_bound_artifact(
        _matmul_definition(),
        trace.workload,
        default_amd_hardware_models()["gfx1200"],
    )
    _ = artifact.to_dict()

    assert trace.model_dump(mode="json") == before


def test_analysis_docs_require_amd_sol_bound_artifact_before_reporting_scores():
    text = (REPO_ROOT / "docs" / "analysis.md").read_text()

    assert "AMD SOL bound artifact" in text
    assert "before reporting AMD-native scores" in text
    assert "supported, inexact, and unsupported" in text


def _cdna3_model(path: Path):
    hardware_path = path / "cdna3-model.json"
    hardware_path.write_text(
        json.dumps(
            {
                "schema_version": "sol_execbench.amd_hardware_model.v2",
                "architecture": "gfx942",
                "dtype_or_path": "bf16/fp32 mixed benchmark path",
                "peak_tflops": 1300.0,
                "memory_bandwidth_gbps": 5300.0,
                "clock_assumptions": ["CDNA3 scaffold for phase 45"],
                "source": "CDNA3 scaffold for phase 45",
                "confidence": "inexact",
                "hardware_validation_status": "unvalidated",
                "model_validation_status": "unvalidated",
                "evidence_refs": ["docs/internal/mi300x_validation_readiness.md"],
            }
        ),
        encoding="utf-8",
    )
    return load_amd_hardware_model(hardware_path)

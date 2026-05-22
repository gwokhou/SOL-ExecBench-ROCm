from __future__ import annotations

from pathlib import Path

from sol_execbench.core.data.definition import Definition
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.scoring.amd_sol import (
    AMD_SOL_SCHEMA_VERSION,
    EstimateConfidence,
    HardwareValidationStatus,
    build_amd_sol_bound_artifact,
    default_amd_hardware_models,
    estimate_work,
    extract_graph,
)


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
    assert payload["hardware_model"]["architecture"] == "gfx1200"
    assert payload["hardware_model"]["validation_status"] == "provisional"


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

    assert graph[0].op_type == "elementwise"
    assert estimates[0].confidence == EstimateConfidence.INEXACT
    assert estimates[0].flops == 16.0
    assert estimates[0].bytes_accessed == 192.0
    assert "one operation per output element" in estimates[0].rationale


def test_unsupported_ops_stay_visible_instead_of_getting_silent_scores():
    definition = Definition(
        name="exp_demo",
        axes={"N": {"type": "var"}},
        inputs={"x": {"shape": ["N"], "dtype": "float32"}},
        outputs={"out": {"shape": ["N"], "dtype": "float32"}},
        reference="import torch\n\ndef run(x):\n    return torch.exp(x)",
    )
    workload = Workload(
        axes={"N": 8},
        inputs={"x": {"type": "random"}},
        uuid="exp-workload",
    )
    hardware = default_amd_hardware_models()["gfx942"]

    artifact = build_amd_sol_bound_artifact(definition, workload, hardware)

    assert artifact.graph_nodes[0].op_type == "unsupported"
    assert artifact.work_estimates[0].confidence == EstimateConfidence.UNSUPPORTED
    assert artifact.op_bounds[0].confidence == EstimateConfidence.UNSUPPORTED
    assert artifact.hardware_model.validation_status == HardwareValidationStatus.UNVALIDATED
    assert artifact.hardware_model.source.endswith("excluded from v1.5")


def test_analysis_docs_require_amd_sol_bound_artifact_before_reporting_scores():
    text = (REPO_ROOT / "docs" / "analysis.md").read_text()

    assert "AMD SOL bound artifact" in text
    assert "before reporting AMD-native scores" in text

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from sol_execbench.core.scoring.amd_hardware_models import amd_hardware_model_from_dict
from sol_execbench.core.scoring.amd_bound_estimate.models import OperatorWorkEstimate
from sol_execbench.core.scoring.amd_bound_graph.models import (
    BoundEdge,
    BoundGraph,
    BoundGraphNode,
    BoundTensor,
    BoundTensorRole,
    OpFamily,
)
from sol_execbench.core.scoring.amd_sol.fusion import build_fusion_groups
from sol_execbench.core.platform.arch_capabilities import (
    load_packaged_arch_capability_budget,
)
from sol_execbench.core.scoring.confidence import EstimateConfidence
from sol_execbench.core.scoring.amd_sol.v3 import (
    AMD_SOL_V3_SCHEMA_VERSION,
    amd_sol_bound_v3_from_dict,
    build_amd_sol_bound_v3_artifact,
)
from sol_execbench.core.scoring.amd_bound_sanity.builder import (
    build_amd_bound_sanity_report,
)
from sol_execbench.core.scoring.amd_score.workload import score_amd_native_workload
from sol_execbench.core.scoring.official_score import (
    BOUND_EVIDENCE_WARNING_BLOCKER,
    official_score_from_amd_native_score,
)
from sol_execbench_type_helpers import make_definition, make_workload


def _hardware():
    return amd_hardware_model_from_dict(
        {
            "schema_version": "sol_execbench.amd_hardware_model.v3",
            "architecture": "gfx1200",
            "clock_assumptions": ["locked"],
            "source": "fixture",
            "confidence": "supported",
            "hardware_validation_status": "validated",
            "model_validation_status": "validated",
            "evidence_refs": ["fixture"],
            "compute_profiles": [
                {
                    "key": "compute.matrix.fp32.fp32.wmma",
                    "state": "measured",
                    "value": 100.0,
                    "confidence": "supported",
                    "evidence_ref": "fixture",
                },
                {
                    "key": "compute.vector.fp32.fp32.portable",
                    "state": "measured",
                    "value": 10.0,
                    "confidence": "supported",
                    "evidence_ref": "fixture",
                },
            ],
            "memory_profiles": [
                {
                    "key": "memory.stream_copy.fp32.fp32.portable",
                    "state": "measured",
                    "value": 100.0,
                    "confidence": "supported",
                    "evidence_ref": "fixture",
                }
            ],
        }
    )


def _definition():
    return make_definition(
        name="fused_matmul_relu",
        axes={
            "M": {"type": "const", "value": 2},
            "K": {"type": "const", "value": 4},
            "N": {"type": "const", "value": 8},
        },
        inputs={
            "a": {"shape": ["M", "K"], "dtype": "float32"},
            "b": {"shape": ["K", "N"], "dtype": "float32"},
        },
        outputs={"out": {"shape": ["M", "N"], "dtype": "float32"}},
        reference="def run(a, b):\n    return (a @ b).relu()\n",
    )


def _workload():
    return make_workload(
        axes={},
        inputs={"a": {"type": "random"}, "b": {"type": "random"}},
        uuid="fused-workload",
    )


def _supported_gfx1200_budget():
    return load_packaged_arch_capability_budget("gfx1200").model_copy(
        update={"confidence": EstimateConfidence.SUPPORTED}
    )


def _exact_fused_definition():
    return make_definition(
        name="fused_matmul_add",
        axes={
            "M": {"type": "const", "value": 2},
            "K": {"type": "const", "value": 4},
            "N": {"type": "const", "value": 8},
        },
        inputs={
            "a": {"shape": ["M", "K"], "dtype": "float32"},
            "b": {"shape": ["K", "N"], "dtype": "float32"},
            "c": {"shape": ["M", "N"], "dtype": "float32"},
        },
        outputs={"out": {"shape": ["M", "N"], "dtype": "float32"}},
        reference="def run(a, b, c):\n    return (a @ b) + c\n",
    )


def _family_epilogue_graph(
    family: OpFamily,
) -> tuple[BoundGraph, tuple[OperatorWorkEstimate, ...]]:
    producer = BoundGraphNode(
        node_id="producer",
        op_family=family,
        op_name=family.value,
        source_expression=family.value,
        input_tensor_ids=("input",),
        output_tensor_ids=("intermediate",),
        attributes={},
        confidence=EstimateConfidence.SUPPORTED,
        rationale="fixture",
    )
    epilogue = BoundGraphNode(
        node_id="epilogue",
        op_family=OpFamily.ELEMENTWISE,
        op_name="add",
        source_expression="producer + bias",
        input_tensor_ids=("intermediate", "bias"),
        output_tensor_ids=("output",),
        attributes={},
        confidence=EstimateConfidence.SUPPORTED,
        rationale="fixture",
    )
    graph = BoundGraph(
        definition="family-epilogue",
        workload_uuid="fixture",
        nodes=(producer, epilogue),
        tensors={
            "input": BoundTensor(
                tensor_id="input",
                name="input",
                role=BoundTensorRole.INPUT,
                shape=(2, 8),
                dtype="float32",
                producer_node_id=None,
                source="fixture",
            ),
            "bias": BoundTensor(
                tensor_id="bias",
                name="bias",
                role=BoundTensorRole.INPUT,
                shape=(2, 8),
                dtype="float32",
                producer_node_id=None,
                source="fixture",
            ),
            "intermediate": BoundTensor(
                tensor_id="intermediate",
                name="intermediate",
                role=BoundTensorRole.INTERMEDIATE,
                shape=(2, 8),
                dtype="float32",
                producer_node_id="producer",
                source="fixture",
            ),
            "output": BoundTensor(
                tensor_id="output",
                name="output",
                role=BoundTensorRole.OUTPUT,
                shape=(2, 8),
                dtype="float32",
                producer_node_id="epilogue",
                source="fixture",
            ),
        },
        edges=(
            BoundEdge(
                edge_id="edge-1",
                source_tensor_id="intermediate",
                target_node_id="epilogue",
                role="input",
            ),
        ),
        warnings=(),
    )
    estimates = tuple(
        OperatorWorkEstimate(
            node_id=node.node_id,
            op_family=node.op_family,
            op_name=node.op_name,
            formula_kind="fixture",
            formula="fixture",
            formula_inputs={},
            flops=1.0,
            read_bytes=64.0,
            write_bytes=64.0,
            intermediate_bytes=0.0,
            movement_bytes=0.0,
            total_bytes=128.0,
            confidence=EstimateConfidence.SUPPORTED,
            rationale="fixture",
        )
        for node in graph.nodes
    )
    return graph, estimates


def test_v3_serializes_a_complete_deterministic_group_partition():
    artifact = build_amd_sol_bound_v3_artifact(
        _definition(),
        _workload(),
        _hardware(),
        hardware_model_ref="fixture-model",
        capability_budget_ref="fixture-budget",
    )
    payload = artifact.to_dict()

    assert payload["schema_version"] == AMD_SOL_V3_SCHEMA_VERSION
    assert payload["capability_budget_ref"] == "fixture-budget"
    assert payload["capability_budget"]["architecture"] == "gfx1200"
    assert {
        node_id for group in artifact.fusion_groups for node_id in group.node_ids
    } == {estimate["node_id"] for estimate in artifact.operator_work_estimates}
    assert sum(len(group.node_ids) for group in artifact.fusion_groups) == len(
        {node_id for group in artifact.fusion_groups for node_id in group.node_ids}
    )
    assert {group.group_id for group in artifact.fusion_groups} == {
        bound.group_id for bound in artifact.group_bounds
    }
    assert all(bound.sol_bound_ms >= 0.0 for bound in artifact.group_bounds)
    assert artifact.aggregate_bound.status == "degraded"
    assert any(warning.startswith("inexact_operator:") for warning in artifact.warnings)
    with pytest.raises(FrozenInstanceError):
        setattr(artifact.fusion_groups[0], "group_id", "changed")


def test_v3_uses_supported_capability_budget_for_exact_gemm_elementwise_fusion():
    artifact = build_amd_sol_bound_v3_artifact(
        _exact_fused_definition(),
        make_workload(
            axes={},
            inputs={
                "a": {"type": "random"},
                "b": {"type": "random"},
                "c": {"type": "random"},
            },
            uuid="fused-exact-workload",
        ),
        _hardware(),
        capability_budget=_supported_gfx1200_budget(),
        capability_budget_ref="validated-budget",
    )

    group = artifact.fusion_groups[0]
    assert group.pattern_id == "gemm_epilogue.v1"
    assert group.required_lds_bytes == 64
    assert "fusion_capacity_evidence_missing" not in group.warnings
    assert group.confidence == EstimateConfidence.SUPPORTED
    assert artifact.group_bounds[0].confidence == EstimateConfidence.SUPPORTED
    assert artifact.aggregate_bound.status == "scored"


def test_v3_rejects_capability_budget_for_a_different_architecture():
    wrong_budget = load_packaged_arch_capability_budget("gfx942")

    with pytest.raises(ValueError, match="must match"):
        build_amd_sol_bound_v3_artifact(
            _definition(),
            _workload(),
            _hardware(),
            capability_budget=wrong_budget,
        )


@pytest.mark.parametrize(
    ("family", "pattern_id"),
    (
        (OpFamily.GEMM, "gemm_epilogue.v1"),
        (OpFamily.LINEAR_PROJECTION, "linear_epilogue.v1"),
        (OpFamily.CONVOLUTION, "convolution_epilogue.v1"),
        (OpFamily.EMBEDDING_POSITIONAL, "embedding_epilogue.v1"),
        (OpFamily.ATTENTION, "attention_epilogue.v1"),
        (OpFamily.REDUCTION, "reduction_epilogue.v1"),
        (OpFamily.NORMALIZATION, "normalization_epilogue.v1"),
        (OpFamily.SOFTMAX, "softmax_epilogue.v1"),
        (OpFamily.MOE, "moe_epilogue.v1"),
        (OpFamily.SSM_MAMBA, "ssm_mamba_epilogue.v1"),
    ),
)
def test_v3_applies_the_same_capacity_checked_epilogue_contract_to_each_family(
    family: OpFamily, pattern_id: str
):
    graph, estimates = _family_epilogue_graph(family)

    group = build_fusion_groups(
        graph, estimates, capability_budget=_supported_gfx1200_budget()
    )[0]

    assert group.pattern_id == pattern_id
    assert group.required_lds_bytes == 64
    assert group.confidence == EstimateConfidence.SUPPORTED


def test_v3_round_trips_and_rejects_tampered_group_partition():
    artifact = build_amd_sol_bound_v3_artifact(_definition(), _workload(), _hardware())
    payload = artifact.to_dict()

    assert amd_sol_bound_v3_from_dict(payload).to_dict() == payload

    duplicate = {
        **payload,
        "fusion_groups": [*payload["fusion_groups"], *payload["fusion_groups"]],
    }
    with pytest.raises(ValueError, match="duplicate group_id"):
        amd_sol_bound_v3_from_dict(duplicate)

    unknown = dict(payload)
    unknown["unexpected"] = True
    with pytest.raises(ValueError, match="unknown field"):
        amd_sol_bound_v3_from_dict(unknown)

    nested_unknown = {
        **payload,
        "aggregate_bound": {**payload["aggregate_bound"], "unexpected": True},
    }
    with pytest.raises(ValueError, match="aggregate_bound has unknown field"):
        amd_sol_bound_v3_from_dict(nested_unknown)


def test_v3_sanity_reports_group_specific_blockers_not_v2_blanket_blocker():
    artifact = build_amd_sol_bound_v3_artifact(_definition(), _workload(), _hardware())

    report = build_amd_bound_sanity_report(amd_sol_artifacts=[artifact.to_dict()])
    blockers = set(report.workloads[0].blocker_codes)

    assert "fusion_semantics_inexact" not in blockers
    assert "fusion_group_inexact" in blockers
    assert any(
        blocker.startswith("fusion_group_warning:fusion_capacity_evidence_missing")
        for blocker in blockers
    )


def test_v3_inexact_fusion_cannot_pass_official_gate():
    artifact = build_amd_sol_bound_v3_artifact(_definition(), _workload(), _hardware())
    score = score_amd_native_workload(
        artifact,
        measured_latency_ms=2.0,
        baseline_latency_ms=3.0,
        trace_ref="trace",
        timing_evidence_ref="timing",
        sol_bound_ref="bound",
        baseline_ref="baseline",
        hardware_model_ref="hardware",
    )

    official = official_score_from_amd_native_score(
        score,
        aggregation_policy="fixed_suite_denominator_zero_for_blocked",
    )

    assert official.status == "blocked"
    assert BOUND_EVIDENCE_WARNING_BLOCKER in official.blocker_reason_codes

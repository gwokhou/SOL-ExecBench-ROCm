from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from solar.analysis.graph_analyzer import (
    EinsumGraphAnalyzer,
    _GraphTopology,
    _PreparedAnalysis,
)
from solar.analysis.graph_models import AnalysisAccumulator, FormalAnalysis, FusionPlan
from solar.analysis.orojenesis import OrojenesisRunner
from solar.rocm.architecture import ArchitectureProfile, MemoryLevel


def _einsum_layer(
    input_name: str,
    weight_name: str,
    output_name: str,
    *,
    m: int = 2,
    k: int = 3,
    n: int = 4,
) -> dict:
    return {
        "type": "matmul",
        "semantic_op": {"kind": "einsum", "equation": "MK,KN->MN", "effects": {}},
        "tensor_names": {
            "inputs": [input_name, weight_name],
            "outputs": [output_name],
        },
        "tensor_shapes": {
            "inputs": [[m, k], [k, n]],
            "outputs": [[m, n]],
        },
        "tensor_dtypes": {
            "inputs": ["torch.float16", "torch.float16"],
            "outputs": ["torch.float16"],
        },
        "connections": {"inputs": [], "outputs": []},
    }


def _start(output_name: str) -> dict:
    return {
        "type": "start",
        "tensor_names": {"inputs": [], "outputs": [output_name]},
        "tensor_dtypes": {"inputs": [], "outputs": ["torch.float16"]},
    }


def _formal_plan() -> tuple[FusionPlan, dict]:
    layers = {
        "c0": _einsum_layer("cx", "cw0", "ch", m=2, k=3, n=4),
        "c1": _einsum_layer("ch", "cw1", "cy", m=2, k=4, n=5),
        "r0": _einsum_layer("rx", "rw0", "rh", m=2, k=3, n=4),
        "r1": _einsum_layer("rh", "rw1", "ry", m=2, k=4, n=6),
        "single": _einsum_layer("sx", "sw", "sy", m=2, k=3, n=4),
    }
    region_problem = {
        "schema_version": 1,
        "kind": "broadcast_batch_linear_matmul",
        "composition": "broadcast_batch_linear_tile_shape_v1",
        "nodes": [
            {"id": "r0", "kind": "matmul", "m": 2, "k": 3, "n": 4, "dtype": "float16"},
            {"id": "r1", "kind": "matmul", "m": 2, "k": 4, "n": 6, "dtype": "float16"},
        ],
        "edges": [
            {
                "producer": "r0",
                "consumer": "r1",
                "axis_map": [0, 1],
                "layer_path": ["r0", "r1"],
            }
        ],
        "roots": ["r0"],
        "leaves": ["r1"],
        "schedule": ["r0", "r1"],
    }
    fusion_regions = [
        {"id": "fusion_chain", "layers": ["c0", "c1"]},
        {"id": "fusion_region", "layers": ["r0", "r1"]},
        {"id": "fusion_single", "layers": ["single"]},
    ]
    plan = FusionPlan(
        fusion={"regions": fusion_regions},
        chains=[["c0", "c1"]],
        regions=[region_problem],
        einsum_layers=layers,
    )
    all_layers = {
        **layers,
        "sx_start": _start("sx"),
        "sw_start": _start("sw"),
    }
    return plan, all_layers


class _FakeRunner:
    toolchain_identity: dict[str, str] | None = {"verification_mode": "fake"}

    @staticmethod
    def _base(problem: dict, word_bits: int) -> dict:
        return {
            "word_bits": word_bits,
            "problem": problem,
            "curve": [
                {"buffer_bytes": 64, "dram_bytes": 160.0},
                {"buffer_bytes": 256, "dram_bytes": 80.0},
            ],
            "evidence_files": {"raw": {"path": "raw.csv", "sha256": "digest"}},
        }

    def run_multi_chain(self, layers, output_dir, *, word_bits):
        del output_dir
        descriptors = []
        dimensions = [(2, 3, 4), (2, 4, 5)]
        for (layer_id, _), (m, k, n) in zip(layers, dimensions):
            descriptors.append({"id": layer_id, "m": m, "k": k, "n": n})
        return self._base({"chain": {"layers": descriptors}}, word_bits)

    def run_multi_region(self, problem, output_dir, *, word_bits):
        del output_dir
        return self._base(problem, word_bits)

    def run_layer(self, layer, output_dir, *, word_bits):
        del layer, output_dir
        return self._base({}, word_bits)


def _empty_orojenesis() -> dict:
    return {
        "status": "not_requested",
        "toolchain": None,
        "layers": {},
        "chains": {},
        "regions": {},
    }


def test_runner_evidence_and_audit_cover_chain_region_and_single(tmp_path: Path):
    analyzer = EinsumGraphAnalyzer()
    plan, all_layers = _formal_plan()
    profile = SimpleNamespace(
        memory_hierarchy=(
            MemoryLevel("l1", "cu", 32),
            MemoryLevel("l2", "device", 128),
            MemoryLevel("vram", "device", 4096),
        )
    )
    prepared = SimpleNamespace(
        output_dir=tmp_path,
        element_size=4.0,
        profile=profile,
        all_layers=all_layers,
    )
    evidence = _empty_orojenesis()
    analyzer._run_orojenesis_evidence(
        plan,
        cast(OrojenesisRunner, _FakeRunner()),
        cast(_PreparedAnalysis, prepared),
        evidence,
        require_orojenesis=True,
    )
    assert evidence["status"] == "complete"
    assert evidence["toolchain"] == {"verification_mode": "fake"}
    assert set(evidence["chains"]) == {"chain_0"}
    assert set(evidence["regions"]) == {"region_0"}
    assert set(evidence["layers"]) == {"single"}
    for category in ("chains", "regions", "layers"):
        for result in evidence[category].values():
            assert result["selected_capacity"]["point"]["buffer_bytes"] == 64
            assert result["evidence_files"]["raw"]["path"].startswith("orojenesis/")

    audited, formal = analyzer._audit_orojenesis_evidence(
        plan,
        evidence,
        cast(_PreparedAnalysis, prepared),
        audited_fused_bytes=48.0,
    )
    assert audited >= 48.0
    assert formal is True
    assert evidence["formal_coverage"] == {"applicable_layers": 5, "total_layers": 5}
    assert evidence["layers"]["single"]["formal_applicability"]["applicable"]
    assert evidence["chains"]["chain_0"]["formal_applicability"]["applicable"]
    assert evidence["regions"]["region_0"]["formal_applicability"]["applicable"]


def test_evidence_selection_handles_no_cache_and_strict_capacity_failure():
    result = {"curve": [{"buffer_bytes": 64, "dram_bytes": 10}], "evidence_files": {}}
    EinsumGraphAnalyzer._select_capacity_and_rewrite_evidence(
        result, None, False, "missing", Path("root")
    )
    assert "selected_capacity" not in result
    with pytest.raises(ValueError, match="missing"):
        EinsumGraphAnalyzer._select_capacity_and_rewrite_evidence(
            result,
            MemoryLevel("tiny", "cu", 1),
            True,
            "missing",
            Path("root"),
        )
    assert EinsumGraphAnalyzer._last_cache(None) is None
    empty = SimpleNamespace(memory_hierarchy=(MemoryLevel("vram", "device", 100),))
    assert EinsumGraphAnalyzer._last_cache(cast(ArchitectureProfile, empty)) is None
    assert EinsumGraphAnalyzer._word_bits([], 4.0) == 32
    assert EinsumGraphAnalyzer._word_bits(["torch.float16", "torch.float32"], 4.0) == 16


def test_strict_runner_requires_toolchain_identity(tmp_path: Path):
    plan, all_layers = _formal_plan()
    runner = _FakeRunner()
    runner.toolchain_identity = None
    prepared = SimpleNamespace(
        output_dir=tmp_path,
        element_size=2.0,
        profile=None,
        all_layers=all_layers,
    )
    with pytest.raises(ValueError, match="toolchain identity"):
        EinsumGraphAnalyzer()._run_orojenesis_evidence(
            plan,
            cast(OrojenesisRunner, runner),
            cast(_PreparedAnalysis, prepared),
            _empty_orojenesis(),
            require_orojenesis=True,
        )


def test_evidence_audits_fail_closed_on_mismatch():
    plan, all_layers = _formal_plan()
    region_by_layer = {
        layer_id: region
        for region in plan.fusion["regions"]
        for layer_id in region["layers"]
    }
    evidence = _empty_orojenesis()
    evidence["layers"]["single"] = {
        "word_bits": 16,
        "selected_capacity": {"point": None},
    }
    assert (
        EinsumGraphAnalyzer._audit_layer_evidence(
            plan, evidence, region_by_layer, all_layers
        )
        == []
    )
    assert not evidence["layers"]["single"]["formal_applicability"]["applicable"]

    evidence["chains"]["bad"] = {
        "word_bits": 16,
        "selected_capacity": {"point": {"dram_bytes": 10}},
        "problem": {"chain": {"layers": [{"id": "missing", "m": 1, "k": 1, "n": 1}]}},
    }
    assert (
        EinsumGraphAnalyzer._audit_chain_evidence(plan, evidence, region_by_layer) == []
    )
    assert not evidence["chains"]["bad"]["formal_applicability"]["applicable"]

    evidence["regions"]["bad"] = {
        "word_bits": 16,
        "selected_capacity": {"point": {"dram_bytes": 10}},
        "problem": {
            "nodes": [{"id": "r0", "m": 1, "k": 1, "n": 1}],
            "roots": ["r0"],
            "leaves": ["r0"],
        },
    }
    assert (
        EinsumGraphAnalyzer._audit_region_evidence(plan, evidence, region_by_layer)
        == []
    )
    assert not evidence["regions"]["bad"]["formal_applicability"]["applicable"]


class _Profile:
    memory_bandwidth_bytes_per_second = 100.0

    @staticmethod
    def resource_seconds(work):
        assert work == {"valu": {"fp32": 2}}
        return {"valu": 2.0, "mfma": 1.0}


def test_lower_bound_combines_compute_and_prefetched_memory():
    prepared = SimpleNamespace(
        profile=_Profile(), semantic_graph=True, semantic_complete=True
    )
    accumulator = SimpleNamespace(resource_work={"valu": {"fp32": 2}})
    formal = FormalAnalysis(None, {}, 50.0, 300.0, True)
    lower = EinsumGraphAnalyzer._lower_bound(
        cast(_PreparedAnalysis, prepared),
        cast(AnalysisAccumulator, accumulator),
        formal,
        require_orojenesis=True,
    )
    assert lower.seconds == 3.0
    assert lower.compute_resource == "valu"
    assert lower.components is not None
    assert lower.components["fused_memory_seconds"] == 0.5

    incomplete = FormalAnalysis(None, {}, 50.0, 50.0, False)
    with pytest.raises(ValueError, match="complete tile-aware"):
        EinsumGraphAnalyzer._lower_bound(
            cast(
                _PreparedAnalysis,
                SimpleNamespace(
                    profile=None, semantic_graph=False, semantic_complete=False
                ),
            ),
            cast(AnalysisAccumulator, accumulator),
            incomplete,
            require_orojenesis=True,
        )


def test_graph_topology_traces_views_consumers_and_orphans():
    layers = {
        "real": {"connections": {"inputs": [], "outputs": ["view"]}},
        "view": {"connections": {"inputs": ["real"], "outputs": ["view2"]}},
        "view2": {"connections": {"inputs": ["view"], "outputs": ["consumer"]}},
        "consumer": {"connections": {"inputs": ["view2"], "outputs": []}},
        "lonely": {"connections": {"inputs": [], "outputs": []}},
    }
    topology = _GraphTopology(
        layers=layers,
        start_node_ids={"start"},
        bool_start_node_ids=set(),
        all_layer_ids=set(layers),
        transparent_layer_ids={"view", "view2", "lonely"},
        tensor_producers={},
        tensor_consumers={},
        intermediate_tensors=set(),
        bool_layers=set(),
        dead_end_layers=set(),
    )
    assert topology.trace_source_through_views("view2") == "real"
    assert topology.trace_source_through_views("lonely") == "lonely"
    assert topology.has_real_consumer("real")
    assert not topology.has_real_consumer("consumer")
    assert not topology.source_is_orphan("view")
    assert not topology.source_is_orphan("start")
    assert topology.source_is_orphan("external")


def test_dequantized_payload_precision_traces_casts_passthrough_and_mul():
    layers = {
        "cast": {
            "semantic_op": {"target": "to"},
            "tensor_names": {"inputs": ["raw"]},
            "tensor_dtypes": {
                "inputs": ["torch.float8_e4m3fn"],
                "outputs": ["torch.float16"],
            },
        },
        "passthrough": {
            "semantic_op": {"target": "view"},
            "tensor_names": {"inputs": ["cast_out"]},
            "tensor_dtypes": {},
        },
        "mul": {
            "semantic_op": {"target": "mul"},
            "tensor_names": {"inputs": ["pass_out", "scale"]},
            "tensor_dtypes": {},
        },
        "same": {
            "semantic_op": {"target": "to"},
            "tensor_names": {"inputs": ["raw"]},
            "tensor_dtypes": {
                "inputs": ["torch.float16"],
                "outputs": ["torch.float16"],
            },
        },
    }
    topology = _GraphTopology(
        layers=layers,
        start_node_ids=set(),
        bool_start_node_ids=set(),
        all_layer_ids=set(layers),
        transparent_layer_ids=set(),
        tensor_producers={
            "cast_out": "cast",
            "pass_out": "passthrough",
            "mul_out": "mul",
            "same_out": "same",
        },
        tensor_consumers={},
        intermediate_tensors=set(),
        bool_layers=set(),
        dead_end_layers=set(),
    )

    class Profile:
        @staticmethod
        def tensor_precision(dtype, fallback):
            del fallback
            return {
                "torch.float8_e4m3fn": "fp8",
                "torch.float16": "fp16",
            }[dtype]

    profile = cast(ArchitectureProfile, Profile())
    assert topology.dequantized_payload_precision("cast_out", profile, "fp32") == "fp8"
    assert topology.dequantized_payload_precision("pass_out", profile, "fp32") == "fp8"
    assert topology.dequantized_payload_precision("mul_out", profile, "fp32") == "fp8"
    assert topology.dequantized_payload_precision("same_out", profile, "fp32") is None
    assert topology.dequantized_payload_precision("missing", profile, "fp32") is None
    assert topology.dequantized_payload_precision("cast_out", None, "fp32") is None
    assert (
        topology.dequantized_payload_precision("cast_out", profile, "fp32", {"cast"})
        is None
    )

"""Frozen top-k gather read provenance on both executable IR paths."""

from pathlib import Path

import pytest
import torch
import yaml

from scripts.internal.solar.base01_records import PROFILE
from solar.analysis.external_reads import external_reads
from solar.analysis.graph_context import build_graph_topology
from solar.artifacts import sha256_file
from solar.contracts import ConversionRequest
from solar.ir.access import AccessQuality, AccessReason
from solar.ir.access_artifact import load_access_document
from solar.ir.contracts import IRPath
from solar.ir.topk_gather import topk_gather_read_proved
from solar.pipeline.readiness import (
    ConversionReadinessRequest,
    audit_conversion,
)

EXPERTS = 256
TOKENS = 64
TOPK = 8


def topk_route(
    gating: torch.Tensor, bias: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Route tokens through a frozen top-k selection and indexed gather."""
    scores = torch.softmax(gating.float(), dim=-1)
    ids = (scores + bias.float()).topk(TOPK, dim=-1, sorted=False).indices
    weights = scores.gather(1, ids) * 2.0
    return weights.float(), ids.to(torch.int32)


def _inputs(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    gating = torch.randn(
        (TOKENS, EXPERTS), generator=generator, dtype=torch.float32
    ).to(torch.bfloat16)
    bias = torch.randn((EXPERTS,), generator=generator, dtype=torch.float32)
    return gating, bias


@pytest.fixture(params=list(IRPath), ids=lambda item: item.value)
def verified_route_graph(request, tmp_path):
    ir_path = request.param
    conversion = ConversionRequest(
        analysis_id="topk-gather",
        reference=topk_route,
        input_factory=_inputs,
        reference_name="test_topk_gather",
        reference_sha256=sha256_file(Path(__file__)),
        ir_path=ir_path,
    )
    output = tmp_path / "conversion"
    result = audit_conversion(
        ConversionReadinessRequest(
            conversion=conversion,
            architecture=PROFILE,
            output_dir=output,
            require_verified_audit=False,
        )
    )
    assert result.ready, result.to_dict()
    return output / ir_path.graph_filename


def test_topk_gather_route_reaches_exact_external_reads(verified_route_graph):
    graph = yaml.safe_load(verified_route_graph.read_text())
    access = load_access_document(verified_route_graph)
    topology = build_graph_topology(
        graph["layers"], set(graph["outputs"]), access
    )
    reads = external_reads(
        topology.layers,
        topology.live_layer_ids,
        set(graph["outputs"]),
        access,
    )
    assert reads.quality is AccessQuality.EXACT
    assert AccessReason.DYNAMIC_INDEX not in reads.reasons
    assert AccessReason.UNKNOWN_READS not in reads.reasons
    report = reads.to_dict()
    assert report["elements"] == TOKENS * EXPERTS + EXPERTS
    assert report["bytes"] == TOKENS * EXPERTS * 2 + EXPERTS * 4


def _gather_layer(
    *,
    dim: int = 1,
    source: str = "scores",
    index: str = "ids",
    source_shape: tuple[int, ...] = (TOKENS, EXPERTS),
    index_shape: tuple[int, ...] = (TOKENS, TOPK),
) -> dict:
    return {
        "semantic_op": {
            "kind": "aten",
            "target": "gather",
            "arguments": [
                {"tensor": 0},
                {"value": dim},
                {"tensor": 1},
            ],
            "kwargs": {},
        },
        "tensor_names": {
            "inputs": [source, index],
            "outputs": ["gathered"],
        },
        "tensor_shapes": {
            "inputs": [list(source_shape), list(index_shape)],
            "outputs": [list(index_shape)],
        },
    }


def _topk_layer(
    *,
    k: object = TOPK,
    dim: object = -1,
    operand: str = "scores",
    shapes: tuple[tuple[int, ...], tuple[int, ...]] = (
        (TOKENS, EXPERTS),
        (TOKENS, TOPK),
    ),
) -> dict:
    arguments: list[object] = [{"tensor": 0}]
    kwargs: dict[str, object] = {}
    if k is not None:
        arguments.append({"value": k})
    if dim is not None:
        arguments.append({"value": dim})
    return {
        "semantic_op": {
            "kind": "aten",
            "target": "topk",
            "arguments": arguments,
            "kwargs": kwargs,
        },
        "tensor_names": {
            "inputs": [operand],
            "outputs": ["values", "ids"],
        },
        "tensor_shapes": {
            "inputs": [list(shapes[0])],
            "outputs": [list(shapes[1]), list(shapes[1])],
        },
    }


def _passthrough_layer(
    name: str, *, source: str, target: str = "identity"
) -> dict:
    return {
        "semantic_op": {"kind": "operation", "target": target, "arguments": []},
        "tensor_names": {"inputs": [source], "outputs": [name]},
        "tensor_shapes": {
            "inputs": [[TOKENS, TOPK]],
            "outputs": [[TOKENS, TOPK]],
        },
    }


def _proved(layers: dict[str, dict], index_name: str = "gathered-ids") -> bool:
    passthrough = _passthrough_layer(index_name, source="ids")
    gather = _gather_layer(index="ids")
    merged = {**layers, "passthrough": passthrough, "gather": gather}
    producer_of = {
        name: layer_id
        for layer_id, layer in merged.items()
        for name in layer["tensor_names"]["outputs"]
    }
    return topk_gather_read_proved(gather, merged, {}, producer_of)


def test_provenance_walks_passthrough_to_topk_index_output():
    assert _proved({"topk": _topk_layer()})


def test_values_output_slot_is_not_index_evidence():
    topk = _topk_layer()
    topk["tensor_names"]["outputs"] = ["ids", "values"]
    assert not _proved({"topk": topk})


def test_non_topk_index_producer_stays_unproved():
    index_source = _topk_layer()
    index_source["semantic_op"] = {
        "kind": "aten",
        "target": "argsort",
        "arguments": [{"tensor": 0}, {"value": -1}],
        "kwargs": {},
    }
    assert not _proved({"argsort": index_source})


def test_dim_and_shape_mismatch_stay_unproved():
    assert not _proved(
        {
            "topk": _topk_layer(
                operand="scores", shapes=((TOKENS, EXPERTS + 8), (TOKENS, TOPK))
            )
        }
    )
    assert not _proved({"topk": _topk_layer(dim=0)})


def test_k_above_extent_and_nonstatic_k_stay_unproved():
    assert not _proved(
        {
            "topk": _topk_layer(
                k=EXPERTS + 1,
                shapes=((TOKENS, EXPERTS), (TOKENS, EXPERTS + 1)),
            )
        }
    )
    assert not _proved({"topk": _topk_layer(k=None)})


def test_gather_over_external_view_stays_unproved():
    passthrough = _passthrough_layer("ids-x", source="ids")
    gather = _gather_layer(source="gating", index="ids-x")
    merged = {
        "topk": _topk_layer(),
        "passthrough": passthrough,
        "gather": gather,
    }
    producer_of = {
        name: layer_id
        for layer_id, layer in merged.items()
        for name in layer["tensor_names"]["outputs"]
    }
    assert not topk_gather_read_proved(
        gather, merged, {"gating": object()}, producer_of
    )

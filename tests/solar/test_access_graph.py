"""Both executable IR dialects derive access maps from structured calls."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml

from scripts.internal.solar.base01_records import PROFILE
from solar.analysis.access_regions import union_accesses
from solar.artifacts import sha256_file
from solar.contracts import ConversionRequest
from solar.ir.access import AccessReason
from solar.ir.access_graph import resolve_graph_accesses
from solar.ir.contracts import IRPath, layer_operation
from solar.pipeline.readiness import (
    ConversionReadinessRequest,
    audit_conversion,
)


@pytest.fixture(params=list(IRPath), ids=[path.value for path in IRPath])
def verified_graph(request, tmp_path):
    ir_path = request.param

    def extract(reference, inputs, *, preserved=()):
        conversion = ConversionRequest(
            analysis_id="access-map",
            reference=reference,
            input_factory=lambda seed: tuple(value.clone() for value in inputs),
            reference_name="test_access_graph",
            reference_sha256=sha256_file(Path(__file__)),
            ir_path=ir_path,
        )
        conversion = replace(
            conversion,
            verification=replace(
                conversion.verification, preserved_input_indices=preserved
            ),
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
        return yaml.safe_load((output / ir_path.graph_filename).read_text())

    return extract


def _reduction_accesses(graph, maps):
    return [
        maps.tensors[layer["tensor_names"]["inputs"][0]]
        for layer in graph["layers"].values()
        if layer_operation(layer).get("target") == "sum"
    ]


@pytest.mark.parametrize(
    ("length", "a", "b", "c", "expected"),
    [
        (128, 64, 32, 96, 96),
        (192, 96, 64, 160, 160),
    ],
    ids=["F04-D", "F04-Q"],
)
def test_overlapping_slice_maps_match_frozen_oracle(
    verified_graph, length, a, b, c, expected
):
    def reference(x):
        return x[:a].sum(), x[b:c].sum()

    graph = verified_graph(
        reference, (torch.arange(length, dtype=torch.float32),)
    )
    for layer in graph["layers"].values():
        layer["raw_attributes"] = (
            "This display string is deliberately unparseable"
        )
    maps = resolve_graph_accesses(graph["layers"])
    assert union_accesses(_reduction_accesses(graph, maps)).elements == expected
    assert not maps.effect_bases


def test_transpose_then_slice_is_in_original_base_coordinates(verified_graph):
    def reference(x):
        return x[:2].sum(), x.transpose(0, 1)[:2].sum()

    graph = verified_graph(reference, (torch.arange(64.0).reshape(8, 8),))
    maps = resolve_graph_accesses(graph["layers"])
    assert union_accesses(_reduction_accesses(graph, maps)).elements == 28


def test_only_consumed_partition_slots_contribute(verified_graph):
    def reference(x):
        a, _, c, _ = x.split(2, dim=0)
        return a.sum(), c.sum()

    graph = verified_graph(reference, (torch.arange(64.0).reshape(8, 8),))
    maps = resolve_graph_accesses(graph["layers"])
    accesses = _reduction_accesses(graph, maps)
    assert union_accesses(accesses).elements == 32
    assert sorted(value.offset for value in accesses) == [0, 32]


def test_different_inputs_are_not_merged_by_matching_shape(verified_graph):
    def reference(x, y):
        return x[:2].sum(), y[:2].sum()

    graph = verified_graph(reference, (torch.ones(4, 8), torch.zeros(4, 8)))
    accesses = _reduction_accesses(
        graph, resolve_graph_accesses(graph["layers"])
    )
    assert len({view.base_id for view in accesses}) == 2
    with pytest.raises(ValueError, match="one base"):
        union_accesses(accesses)


def test_clone_storage_is_fresh_and_not_transparent(verified_graph):
    def reference(x):
        return x.clone()[:2].sum()

    graph = verified_graph(reference, (torch.ones(4, 8),))
    maps = resolve_graph_accesses(graph["layers"])
    for layer_id, layer in graph["layers"].items():
        if layer_operation(layer).get("target") == "clone":
            assert layer_id not in maps.transparent_layers
            # Batch-2: clone copies into fresh storage; its output reason is
            # MATERIALIZED so downstream views resolve instead of inheriting
            # copy_boundary.
            assert all(
                maps.reasons[name] is AccessReason.MATERIALIZED
                for name in layer["tensor_names"]["outputs"]
            )


def test_inconsistent_view_metadata_rejected_before_accounting(verified_graph):
    def reference(x):
        return x[2:6].sum()

    graph = verified_graph(reference, (torch.ones(8),))
    damaged = deepcopy(graph)
    for layer in damaged["layers"].values():
        if layer_operation(layer).get("target") in {"slice", "getitem"}:
            layer["tensor_shapes"]["outputs"] = [[5]]
            break
    with pytest.raises(ValueError, match="shape disagrees"):
        resolve_graph_accesses(damaged["layers"])


def test_contiguous_view_reshape_composition(verified_graph):
    def reference(x):
        return x.view(4, 4)[1:3].sum()

    graph = verified_graph(reference, (torch.arange(16.0),))
    maps = resolve_graph_accesses(graph["layers"])
    assert union_accesses(_reduction_accesses(graph, maps)).elements == 8


def test_dynamic_indices_do_not_acquire_a_static_exact_map(verified_graph):
    def reference(x, indices):
        return x.index_select(0, indices)

    graph = verified_graph(
        reference,
        (torch.arange(32.0).reshape(8, 4), torch.tensor([0, 7, 7])),
        preserved=(1,),
    )
    maps = resolve_graph_accesses(graph["layers"])
    for name in graph["outputs"]:
        assert name not in maps.tensors
        assert maps.reasons[name] is AccessReason.DYNAMIC_INDEX


def test_nonunit_slice_retains_an_explicit_unavailable_reason(verified_graph):
    def reference(x):
        return x[1:12:2]

    graph = verified_graph(reference, (torch.arange(16.0),))
    maps = resolve_graph_accesses(graph["layers"])
    for name in graph["outputs"]:
        assert name not in maps.tensors
        assert maps.reasons[name] is AccessReason.NONUNIT_STEP


def test_contiguous_reshape_input_proves_the_conditional_alias(verified_graph):
    # Batch-2: a contiguous input proves the conditional alias as a view.
    def reference(x):
        return x.reshape(4, 4)

    graph = verified_graph(reference, (torch.arange(16.0),))
    maps = resolve_graph_accesses(graph["layers"])
    for layer in graph["layers"].values():
        if layer_operation(layer).get("target") in {"reshape", "view"}:
            outputs = layer["tensor_names"]["outputs"]
            assert all(name in maps.tensors for name in outputs)


def test_noncontiguous_reshape_input_copies_into_materialized_storage(
    verified_graph,
    request,
):
    # Batch-2: a non-contiguous input forces a copy into fresh storage.
    if "torchview" in request.node.callspec.id:
        pytest.skip(
            "torchview cannot convert the transposed-reshape composition"
        )

    def reference(x):
        return x.t().reshape(8, 4)

    graph = verified_graph(reference, (torch.arange(32.0).reshape(4, 8),))
    maps = resolve_graph_accesses(graph["layers"])
    for layer_id, layer in graph["layers"].items():
        if layer_operation(layer).get("target") == "reshape":
            assert layer_id not in maps.transparent_layers
            for name in layer["tensor_names"]["outputs"]:
                assert maps.reasons[name] is AccessReason.MATERIALIZED

"""Effects that replay observes must not disappear from cost accounting."""

from pathlib import Path

import pytest

from scripts.internal.solar.base01_oracles import resource_expectations
from scripts.internal.solar.base01_records import PROFILE
from scripts.internal.solar.base01_samples import get_sample
from solar.analysis.graph_context import build_graph_topology
from solar.artifacts import sha256_file
from solar.contracts import AnalysisRequest
from solar.ir.access_graph import resolve_graph_accesses
from solar.ir.contracts import IRGraphArtifact, IRPath
from solar.pipeline.readiness import (
    ConversionReadinessRequest,
    audit_conversion,
)
from solar.pipeline.stages import analyze_request_graph
from solar.rocm.architecture import ArchitectureProfile


def _layer(name, source, *, alias=False, mutates=False, opaque=False):
    return {
        "type": "add" if mutates else "view" if alias else "clone",
        "tensor_names": {"inputs": [source], "outputs": [name]},
        "connections": {"inputs": [source], "outputs": []},
        "semantic_op": {
            "kind": "operation",
            "target": "add" if mutates else "view" if alias else "clone",
            "effects": {
                "mutates": [0] if mutates else [],
                "aliases": [{"input": 0, "output": 0}]
                if alias or mutates
                else [],
                "atomic": False,
                "opaque_library_call": opaque,
            },
        },
    }


def test_external_mutation_is_live_even_when_another_alias_is_returned():
    layers = {
        "write_view": _layer("write_view", "input", alias=True),
        "read_view": _layer("read_view", "input", alias=True),
        "write": _layer("write", "write_view", mutates=True),
    }
    topology = build_graph_topology(layers, {"read_view"})
    assert topology.live_layer_ids == set(layers)


@pytest.mark.parametrize(
    "returned", ["copy", "alias"], ids=["storage", "alias"]
)
def test_mutation_of_returned_local_storage_is_live(returned):
    layers = {
        "copy": _layer("copy", "input"),
        "alias": _layer("alias", "copy", alias=True),
        "write": _layer("write", "copy", mutates=True),
    }
    topology = build_graph_topology(layers, {returned})
    assert {"copy", "write"} <= topology.live_layer_ids


def test_unobserved_local_mutation_is_dead_and_copy_stops_alias_traversal():
    layers = {
        "copy": _layer("copy", "input"),
        "write": _layer("write", "copy", mutates=True),
        "result": _layer("result", "input", alias=True),
    }
    topology = build_graph_topology(layers, {"result"})
    assert topology.live_layer_ids == {"result"}


def test_conditional_alias_does_not_erase_a_potentially_observable_write():
    layers = {
        "reshape": _layer("reshape", "input", alias=True),
        "write": _layer("write", "reshape", mutates=True),
        "result": _layer("result", "input", alias=True),
    }
    layers["reshape"]["semantic_op"]["effects"]["aliases"][0]["conditional"] = (
        True
    )
    assert build_graph_topology(layers, {"result"}).live_layer_ids == set(
        layers
    )


def test_opaque_effect_cannot_be_removed_by_return_value_liveness():
    layers = {
        "call": _layer("call", "input", opaque=True),
        "result": _layer("result", "input", alias=True),
    }
    assert build_graph_topology(layers, {"result"}).live_layer_ids == set(
        layers
    )


def test_atomic_mutation_to_external_storage_remains_live():
    layers = {
        "atomic": _layer("atomic", "input", mutates=True),
        "result": _layer("result", "input", alias=True),
    }
    layers["atomic"]["semantic_op"]["effects"]["atomic"] = True
    assert build_graph_topology(layers, {"result"}).live_layer_ids == set(
        layers
    )


@pytest.mark.parametrize(
    "ir_path", list(IRPath), ids=[path.value for path in IRPath]
)
@pytest.mark.parametrize("sample_id", ["F08-D", "F08-Q"])
def test_verified_mutation_cost_matches_independent_base01_oracle(
    tmp_path: Path, ir_path: IRPath, sample_id: str
) -> None:
    sample = get_sample(sample_id)
    conversion = sample.request(
        ir_path, sha256_file(Path(get_sample.__code__.co_filename))
    )
    converted = tmp_path / "conversion"
    readiness = audit_conversion(
        ConversionReadinessRequest(
            conversion=conversion,
            architecture=PROFILE,
            output_dir=converted,
            require_verified_audit=False,
        )
    )
    assert readiness.ready
    request = AnalysisRequest(
        conversion=conversion,
        architecture=PROFILE,
        output_dir=tmp_path,
        precision="fp32",
        require_orojenesis=False,
        require_verified_audit=False,
    )
    result = analyze_request_graph(
        request,
        ArchitectureProfile.load(PROFILE),
        tmp_path,
        IRGraphArtifact(
            path=converted / ir_path.graph_filename,
            kind=ir_path.ir_kind,
        ),
    )
    assert (
        result["total"]["resource_work"] == resource_expectations()[sample_id]
    )
    assert result["total"]["num_orphaned_layers"] == 0
    graph = IRGraphArtifact(
        path=converted / ir_path.graph_filename,
        kind=ir_path.ir_kind,
    )
    layers = {}
    for name, layer in graph.document.require_mapping("layers").items():
        assert isinstance(layer, dict)
        layers[name] = layer
    maps = resolve_graph_accesses(layers)
    assert len(maps.effect_bases) == 1


@pytest.mark.parametrize("write_before_read", [True, False])
def test_local_write_is_live_only_before_a_live_read(write_before_read):
    layers = {
        "copy": _layer("copy", "input"),
        "alias": _layer("alias", "copy", alias=True),
    }
    write = _layer("write", "copy", mutates=True)
    read = _layer("result", "alias")
    layers.update(
        {"write": write, "result": read}
        if write_before_read
        else {"result": read, "write": write}
    )
    expected = {"copy", "alias", "result"}
    if write_before_read:
        expected.add("write")
    assert build_graph_topology(layers, {"result"}).live_layer_ids == expected


def test_effect_read_can_make_an_earlier_local_effect_live():
    layers = {
        "copy": _layer("copy", "input"),
        "alias": _layer("alias", "copy", alias=True),
        "write": _layer("write", "copy", mutates=True),
        "external_write": _layer("external_write", "input", mutates=True),
        "result": _layer("result", "input", alias=True),
    }
    layers["external_write"]["tensor_names"]["inputs"].append("alias")
    layers["external_write"]["connections"]["inputs"].append("alias")
    assert build_graph_topology(layers, {"result"}).live_layer_ids == set(
        layers
    )


@pytest.mark.parametrize(
    "order",
    [
        ("copy", "alias", "write", "result"),
        ("result", "alias", "copy", "write"),
    ],
)
def test_local_effect_follows_replay_order_not_mapping_order(order):
    layers = {
        "copy": _layer("copy", "input"),
        "alias": _layer("alias", "copy", alias=True),
        "write": _layer("write", "copy", mutates=True),
        "result": _layer("result", "alias"),
    }
    ordered = {name: layers[name] for name in order}
    assert build_graph_topology(ordered, {"result"}).live_layer_ids == set(
        layers
    )

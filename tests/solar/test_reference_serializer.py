from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from solar.graph.reference_serializer import ReferenceGraphSerializer
from solar.schema_versions import OPERATOR_GRAPH_SCHEMA_VERSION


def test_make_fx_reference_records_strict_conversion_provenance() -> None:
    from torch.fx.experimental.proxy_tensor import make_fx

    graph = make_fx(lambda value: torch.sin(value))(torch.ones(2))
    result = ReferenceGraphSerializer().serialize_fx_reference(
        graph,
        "reference",
    )

    assert result["schema_version"] == OPERATOR_GRAPH_SCHEMA_VERSION
    assert result["extraction_kind"] == "make_fx_reference_v1"
    assert result["joint_graph"] is False
    assert any(
        layer.get("phase") == "reference" for layer in result["layers"].values()
    )
    assert all(
        layer.get("phase") in {"input", "reference"}
        for layer in result["layers"].values()
    )


def test_canonical_target_retains_exact_aten_target_and_overload() -> None:
    from torch.fx.experimental.proxy_tensor import make_fx

    graph = make_fx(lambda value: value.to(torch.float16))(torch.ones(2))
    result = ReferenceGraphSerializer().serialize_fx_reference(
        graph,
        "conversion",
    )
    conversion = next(
        layer
        for layer in result["layers"].values()
        if (layer.get("semantic_op") or {}).get("target") == "to"
    )

    assert conversion["semantic_op"]["exact_target"] == "_to_copy"
    assert conversion["semantic_op"]["overload"] == "default"


def test_argument_and_tensor_metadata_serialization() -> None:
    graph = torch.fx.Graph()
    node = graph.placeholder("value")
    inputs = [node]

    assert ReferenceGraphSerializer._serialize_argument(node, inputs) == {
        "tensor": 0,
    }
    assert ReferenceGraphSerializer._serialize_argument(
        (
            torch.float16,
            torch.device("cpu"),
            torch.strided,
            None,
            [2, "x"],
        ),
        inputs,
    ) == [
        {"dtype": "float16"},
        {"device": "cpu"},
        {"layout": "strided"},
        {"value": None},
        [{"value": 2}, {"value": "x"}],
    ]
    assert ReferenceGraphSerializer._serialize_argument(
        torch.preserve_format,
        inputs,
    ) == ("preserve_format")
    assert ReferenceGraphSerializer._serialize_argument(
        torch.contiguous_format,
        inputs,
    ) == ("contiguous_format")
    assert ReferenceGraphSerializer._serialize_argument(object(), inputs)[
        "value"
    ].startswith("<object object")
    assert ReferenceGraphSerializer._tensor_metadata(
        (torch.zeros(2, dtype=torch.float16), [torch.ones(1), "ignored"]),
    ) == [([2], "torch.float16"), ([1], "torch.float32")]
    assert ReferenceGraphSerializer._tensor_metadata("ignored") == []


def test_schema_effects_reports_mutation_alias_and_atomicity() -> None:
    graph = torch.fx.Graph()
    left = graph.placeholder("left")
    right = graph.placeholder("right")
    mutation = graph.call_function(torch.ops.aten.add_.Tensor, (left, right))

    effects = ReferenceGraphSerializer._schema_effects(
        mutation,
        [left, right],
        target_name="add_",
        exact_target="add",
        output_arity=1,
    )

    assert effects["mutates"] == [0]
    assert effects["aliases"] == [{"output": 0, "input": 0}]
    assert effects["atomic"] is False

    scatter = graph.call_function(
        torch.ops.aten.scatter.src,
        (left, 0, right, right),
    )
    atomic = ReferenceGraphSerializer._schema_effects(
        scatter,
        [left, right],
        target_name="scatter",
        exact_target="scatter",
        output_arity=1,
    )
    assert atomic["atomic"] is True

    split = graph.call_function(
        torch.ops.aten.split_with_sizes.default,
        (left, [1, 1], 0),
    )
    split_effects = ReferenceGraphSerializer._schema_effects(
        split,
        [left],
        target_name="split_with_sizes",
        exact_target="aten.split_with_sizes.default",
        output_arity=2,
    )
    assert split_effects["aliases"] == [
        {"output": 0, "input": 0},
        {"output": 1, "input": 0},
    ]


def test_schema_effects_rejects_missing_or_inconsistent_schema() -> None:
    target = SimpleNamespace()
    node = SimpleNamespace(target=target, args=(), kwargs={})
    with pytest.raises(RuntimeError, match="has no FunctionSchema"):
        ReferenceGraphSerializer._schema_effects(
            node,
            [],
            target_name="opaque",
            exact_target="opaque",
            output_arity=1,
        )

    graph = torch.fx.Graph()
    left = graph.placeholder("left")
    right = graph.placeholder("right")
    pure = graph.call_function(torch.ops.aten.add.Tensor, (left, right))
    with pytest.raises(RuntimeError, match="lacks a schema write effect"):
        ReferenceGraphSerializer._schema_effects(
            pure,
            [left, right],
            target_name="add_",
            exact_target="add",
            output_arity=1,
        )

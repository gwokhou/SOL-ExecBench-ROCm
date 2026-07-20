from __future__ import annotations

from copy import deepcopy

import pytest

from solar.einsum import semantics
from solar.einsum.semantics import SemanticGraphError


def _layer(operation: str = "add", *, inputs: int = 2, outputs: int = 1):
    return {
        "type": operation,
        "is_real_einsum": False,
        "tensor_names": {
            "inputs": [f"input_{index}" for index in range(inputs)],
            "outputs": [f"output_{index}" for index in range(outputs)],
        },
        "tensor_shapes": {
            "inputs": [[2] for _ in range(inputs)],
            "outputs": [[2] for _ in range(outputs)],
        },
        "tensor_dtypes": {
            "inputs": ["torch.float32" for _ in range(inputs)],
            "outputs": ["torch.float32" for _ in range(outputs)],
        },
    }


def _graph(layer=None):
    return {
        "schema_version": semantics.EINSUM_GRAPH_SCHEMA_VERSION,
        "layers": {"operation": layer or _layer()},
    }


@pytest.mark.parametrize(
    ("operation", "target"),
    [
        ("torch.concat", "cat"),
        ("convtranspose2d", "conv_transpose2d"),
        ("Tensor.__getitem__", "getitem"),
        ("max", "amax"),
        ("min", "amin"),
        ("t", "transpose"),
        ("type", "to"),
        ("sdpa", "scaled_dot_product_attention"),
        ("add_", "add"),
    ],
)
def test_build_semantics_canonicalizes_operation_names(operation, target) -> None:
    semantic = semantics.build_semantic_operation(_layer(operation))
    assert semantic["target"] == target


def test_build_semantics_supports_inputs_einsum_and_recorded_values() -> None:
    start = _layer("start", inputs=0)
    assert semantics.build_semantic_operation(start) == {
        "kind": "input",
        "target": "input",
        "arguments": [],
        "kwargs": {},
    }

    einsum = _layer("matmul")
    einsum.update(is_real_einsum=True, einsum_equation="AB,BC->AC")
    semantic = semantics.build_semantic_operation(einsum)
    assert semantic["kind"] == "einsum"
    assert semantic["equation"] == "AB,BC->AC"

    recorded = _layer("sum", inputs=1)
    recorded["module_args"] = {
        "call_arguments": [{"tensor": 0}],
        "call_kwargs": {"dim": [0], "keepdim": True, "raw_attributes": "drop"},
    }
    semantic = semantics.build_semantic_operation(recorded)
    assert semantic["arguments"] == [{"tensor": 0}]
    assert semantic["kwargs"] == {"dim": [0], "keepdim": True}


def test_reverse_operations_swap_arguments_and_validate_arity() -> None:
    reverse = _layer("__rsub__")
    reverse["module_args"] = {
        "call_arguments": [{"tensor": 0}, {"tensor": 1}],
        "call_kwargs": {},
    }
    semantic = semantics.build_semantic_operation(reverse)
    assert semantic["target"] == "sub"
    assert semantic["arguments"] == [{"tensor": 1}, {"tensor": 0}]

    reverse["module_args"]["call_arguments"] = [{"tensor": 0}]
    with pytest.raises(SemanticGraphError, match="requires exactly two"):
        semantics.build_semantic_operation(reverse)


def test_inferred_kwargs_views_and_effects_are_explicit() -> None:
    reshape = _layer("reshape", inputs=1)
    reshape["tensor_shapes"]["outputs"] = [[1, 2]]
    reshape["module_args"] = {"dims": [0], "training": True}
    semantic = semantics.build_semantic_operation(reshape)
    assert semantic["kwargs"] == {"dim": [0], "shape": [1, 2]}
    assert semantic["effects"]["aliases"] == [
        {"output": 0, "input": 0, "conditional": True}
    ]

    mutation = _layer("copy_", inputs=1)
    mutation["aliases"] = [{"output": 0, "input": 0}]
    assert semantics.build_semantic_operation(mutation)["effects"]["mutates"] == [0]

    scatter = _layer("scatter", inputs=3)
    assert semantics.build_semantic_operation(scatter)["effects"]["atomic"] is True
    linear = _layer("linear", inputs=3)
    assert (
        semantics.build_semantic_operation(linear)["effects"]["opaque_library_call"]
        is True
    )


def test_softmax_requires_dimension_and_non_strict_annotation_records_unsupported() -> (
    None
):
    graph = {"layers": {"softmax": _layer("softmax", inputs=1)}}
    annotated = semantics.annotate_semantics(deepcopy(graph), strict=False)
    assert annotated["layers"]["softmax"]["semantic_op"]["kind"] == "unsupported"

    with pytest.raises(SemanticGraphError, match="explicit dim"):
        semantics.annotate_semantics(deepcopy(graph), strict=True)


def test_annotation_rejects_non_mapping_layer() -> None:
    with pytest.raises(SemanticGraphError, match="is not a mapping"):
        semantics.annotate_semantics({"layers": {"bad": []}}, strict=False)


def test_validate_accepts_input_einsum_and_dynamic_aten_targets() -> None:
    start = _layer("start", inputs=0)
    start["semantic_op"] = semantics.build_semantic_operation(start)
    einsum = _layer("matmul")
    einsum.update(is_real_einsum=True, einsum_equation="A,A->A")
    einsum["semantic_op"] = semantics.build_semantic_operation(einsum)
    dynamic = _layer("sin", inputs=1)
    dynamic["semantic_op"] = {
        "kind": "aten",
        "target": "sin",
        "arguments": [{"tensor": 0}],
        "kwargs": {},
        "effects": {"mutates": [], "aliases": [], "atomic": False},
    }
    graph = {
        "schema_version": 3,
        "layers": {"start": start, "einsum": einsum, "dynamic": dynamic},
    }
    semantics.validate_semantic_graph(graph)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda graph: graph.update(schema_version=2), "latest schema_version"),
        (lambda graph: graph.update(layers={}), "has no layers"),
        (lambda graph: graph["layers"].update(operation=[]), "is not a mapping"),
        (
            lambda graph: graph["layers"]["operation"].update(
                tensor_dtypes={"inputs": [], "outputs": ["torch.float32"]}
            ),
            "explicit inputs name/shape/dtype",
        ),
        (
            lambda graph: graph["layers"]["operation"].pop("semantic_op"),
            "has no semantic_op",
        ),
        (
            lambda graph: graph["layers"]["operation"]["semantic_op"].update(
                kind="unsupported"
            ),
            "not executable exactly",
        ),
        (
            lambda graph: graph["layers"]["operation"]["semantic_op"].update(
                target="not-a-real-operation"
            ),
            "unsupported exact operation",
        ),
        (
            lambda graph: graph["layers"]["operation"]["semantic_op"].update(
                arguments={}
            ),
            "lacks explicit arguments",
        ),
        (
            lambda graph: graph["layers"]["operation"]["semantic_op"].update(
                kwargs=["bad"]
            ),
            "invalid keyword arguments",
        ),
        (
            lambda graph: graph["layers"]["operation"]["semantic_op"].update(
                arguments=[{"tensor": 4}, {"tensor": 1}]
            ),
            "outside its input metadata",
        ),
        (
            lambda graph: graph["layers"]["operation"]["semantic_op"].update(
                arguments=[{"tensor": 0}, {"value": 1}]
            ),
            "does not preserve every ordered tensor",
        ),
        (
            lambda graph: graph["layers"]["operation"]["semantic_op"].pop("effects"),
            "lacks explicit effects",
        ),
        (
            lambda graph: graph["layers"]["operation"]["semantic_op"]["effects"].update(
                mutates="bad"
            ),
            "invalid mutation/alias effects",
        ),
        (
            lambda graph: graph["layers"]["operation"]["semantic_op"]["effects"].update(
                mutates=[4]
            ),
            "invalid mutation target",
        ),
        (
            lambda graph: graph["layers"]["operation"]["semantic_op"]["effects"].update(
                aliases=[{"input": 0, "output": 4}]
            ),
            "invalid alias effect",
        ),
    ],
)
def test_validate_rejects_incomplete_semantic_contracts(mutate, message) -> None:
    graph = _graph()
    graph["layers"]["operation"]["semantic_op"] = semantics.build_semantic_operation(
        graph["layers"]["operation"]
    )
    mutate(graph)
    with pytest.raises(SemanticGraphError, match=message):
        semantics.validate_semantic_graph(graph)


def test_validate_rejects_bad_einsum_and_missing_required_parameters() -> None:
    einsum = _layer("matmul")
    einsum["semantic_op"] = {"kind": "einsum", "equation": "AB"}
    with pytest.raises(SemanticGraphError, match="no exact einsum equation"):
        semantics.validate_semantic_graph(_graph(einsum))

    softmax = _layer("softmax", inputs=1)
    softmax["semantic_op"] = {
        "kind": "aten",
        "target": "softmax",
        "arguments": [{"tensor": 0}],
        "kwargs": {},
        "effects": {"mutates": [], "aliases": []},
    }
    with pytest.raises(SemanticGraphError, match="lacks exact softmax parameters"):
        semantics.validate_semantic_graph(_graph(softmax))

    sliced = _layer("slice", inputs=1)
    sliced["semantic_op"] = {
        "kind": "aten",
        "target": "slice",
        "arguments": [{"tensor": 0}],
        "kwargs": {"dim": 0},
        "effects": {"mutates": [], "aliases": []},
    }
    with pytest.raises(SemanticGraphError, match="explicit slice bounds"):
        semantics.validate_semantic_graph(_graph(sliced))

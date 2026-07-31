from __future__ import annotations

import pytest

from solar.ir.aten.conversion import AtenIRError, validate_aten_graph
from solar.schema_versions import ATEN_IR_SCHEMA_VERSION


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
        "schema_version": ATEN_IR_SCHEMA_VERSION,
        "ir_kind": "aten",
        "layers": {"operation": layer or _layer()},
    }


def _aten_semantic(
    target: str = "add",
    input_count: int = 2,
) -> dict[str, object]:
    return {
        "kind": "aten",
        "target": target,
        "arguments": [{"tensor": index} for index in range(input_count)],
        "kwargs": {},
        "effects": {"mutates": [], "aliases": [], "atomic": False},
    }


def test_validate_accepts_input_einsum_and_dynamic_aten_targets() -> None:
    start = _layer("start", inputs=0)
    start["semantic_op"] = {
        "kind": "input",
        "target": "input",
        "arguments": [],
        "kwargs": {},
    }
    einsum = _layer("matmul")
    einsum.update(is_real_einsum=True, einsum_equation="A,A->A")
    einsum["semantic_op"] = {
        "kind": "einsum",
        "target": "einsum",
        "equation": "A,A->A",
        "arguments": [{"tensor": 0}, {"tensor": 1}],
        "kwargs": {},
        "effects": {"mutates": [], "aliases": [], "atomic": False},
    }
    dynamic = _layer("sin", inputs=1)
    dynamic["semantic_op"] = {
        "kind": "aten",
        "target": "sin",
        "arguments": [{"tensor": 0}],
        "kwargs": {},
        "effects": {"mutates": [], "aliases": [], "atomic": False},
    }
    graph = {
        "schema_version": ATEN_IR_SCHEMA_VERSION,
        "ir_kind": "aten",
        "layers": {"start": start, "einsum": einsum, "dynamic": dynamic},
    }
    validate_aten_graph(graph)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda graph: graph.update(ir_kind="extended_einsum"),
            "not ATen IR",
        ),
        pytest.param(
            lambda graph: graph.update(schema_version=0),
            "current schema_version",
            id="numeric-schema-version",
        ),
        pytest.param(
            lambda graph: graph.update(
                schema_version=str(ATEN_IR_SCHEMA_VERSION),
            ),
            "current schema_version",
            id="string-schema-version",
        ),
        (lambda graph: graph.update(layers={}), "has no layers"),
        (
            lambda graph: graph["layers"].update(operation=[]),
            "is not a mapping",
        ),
        (
            lambda graph: graph["layers"]["operation"].update(
                tensor_dtypes={"inputs": [], "outputs": ["torch.float32"]},
            ),
            "explicit inputs name/shape/dtype",
        ),
        (
            lambda graph: graph["layers"]["operation"].pop("semantic_op"),
            "has no semantic_op",
        ),
        (
            lambda graph: graph["layers"]["operation"]["semantic_op"].update(
                kind="unsupported",
            ),
            "not executable exactly",
        ),
        (
            lambda graph: graph["layers"]["operation"]["semantic_op"].update(
                target="not-a-real-operation",
            ),
            "unsupported exact operation",
        ),
        (
            lambda graph: graph["layers"]["operation"]["semantic_op"].update(
                arguments={},
            ),
            "lacks explicit arguments",
        ),
        (
            lambda graph: graph["layers"]["operation"]["semantic_op"].update(
                kwargs=["bad"],
            ),
            "invalid keyword arguments",
        ),
        (
            lambda graph: graph["layers"]["operation"]["semantic_op"].update(
                arguments=[{"tensor": 4}, {"tensor": 1}],
            ),
            "outside its input metadata",
        ),
        (
            lambda graph: graph["layers"]["operation"]["semantic_op"].update(
                arguments=[{"tensor": 0}, {"value": 1}],
            ),
            "does not preserve every ordered tensor",
        ),
        (
            lambda graph: graph["layers"]["operation"]["semantic_op"].pop(
                "effects",
            ),
            "lacks explicit effects",
        ),
        (
            lambda graph: graph["layers"]["operation"]["semantic_op"][
                "effects"
            ].update(
                mutates="bad",
            ),
            "invalid mutation/alias effects",
        ),
        (
            lambda graph: graph["layers"]["operation"]["semantic_op"][
                "effects"
            ].update(
                mutates=[4],
            ),
            "invalid mutation target",
        ),
        (
            lambda graph: graph["layers"]["operation"]["semantic_op"][
                "effects"
            ].update(
                aliases=[{"input": 0, "output": 4}],
            ),
            "invalid alias effect",
        ),
    ],
)
def test_validate_rejects_incomplete_aten_contracts(
    mutate,
    message,
) -> None:
    graph = _graph()
    graph["layers"]["operation"]["semantic_op"] = _aten_semantic()
    mutate(graph)
    with pytest.raises(AtenIRError, match=message):
        validate_aten_graph(graph)


def test_validate_rejects_bad_einsum_and_missing_required_parameters() -> None:
    einsum = _layer("matmul")
    einsum["semantic_op"] = {"kind": "einsum", "equation": "AB"}
    with pytest.raises(AtenIRError, match="no exact einsum equation"):
        validate_aten_graph(_graph(einsum))

    softmax = _layer("softmax", inputs=1)
    softmax["semantic_op"] = {
        "kind": "aten",
        "target": "softmax",
        "arguments": [{"tensor": 0}],
        "kwargs": {},
        "effects": {"mutates": [], "aliases": []},
    }
    with pytest.raises(AtenIRError, match="lacks exact softmax parameters"):
        validate_aten_graph(_graph(softmax))

    sliced = _layer("slice", inputs=1)
    sliced["semantic_op"] = {
        "kind": "aten",
        "target": "slice",
        "arguments": [{"tensor": 0}],
        "kwargs": {"dim": 0},
        "effects": {"mutates": [], "aliases": []},
    }
    with pytest.raises(AtenIRError, match="explicit slice bounds"):
        validate_aten_graph(_graph(sliced))


def test_validate_accepts_positional_slice_parameters() -> None:
    sliced = _layer("slice", inputs=1)
    sliced["semantic_op"] = {
        "kind": "aten",
        "target": "slice",
        "exact_target": "aten.slice.Tensor",
        "overload": "Tensor",
        "arguments": [
            {"tensor": 0},
            {"value": 1},
            {"value": 0},
            {"value": 2},
            {"value": 1},
        ],
        "kwargs": {},
        "effects": {
            "mutates": [],
            "aliases": [{"input": 0, "output": 0}],
            "atomic": False,
        },
    }

    validate_aten_graph(_graph(sliced))


def test_validate_requires_topk_parameters_and_two_output_slots() -> None:
    topk = _layer("topk", inputs=1)
    topk["semantic_op"] = {
        "kind": "aten",
        "target": "topk",
        "exact_target": "aten.topk.default",
        "overload": "default",
        "arguments": [{"tensor": 0}, {"value": 2}],
        "kwargs": {"dim": -1},
        "effects": {"mutates": [], "aliases": [], "atomic": False},
    }

    with pytest.raises(AtenIRError, match="two output slots"):
        validate_aten_graph(_graph(topk))

    topk["tensor_names"]["outputs"] = ["values", "indices"]
    topk["tensor_shapes"]["outputs"] = [[2], [2]]
    topk["tensor_dtypes"]["outputs"] = ["torch.float32", "torch.int64"]
    validate_aten_graph(_graph(topk))

from __future__ import annotations

import pytest

from solar.analysis.contraction_proofs import (
    build_orojenesis_proof_layer,
    requires_tile_evidence,
)
from solar.ir.extended_einsum.operations.analyzer import EinsumAnalyzer


def _aten_layer(
    target: str,
    input_shapes: list[list[int]],
    output_shape: list[int],
    *,
    arguments: list[object] | None = None,
) -> dict:
    return {
        "type": target,
        "semantic_op": {
            "kind": "aten",
            "target": target,
            "arguments": arguments or [],
            "effects": {
                "mutates": [],
                "aliases": [],
                "atomic": False,
                "opaque_library_call": False,
            },
        },
        "tensor_names": {
            "inputs": [f"input_{index}" for index in range(len(input_shapes))],
            "outputs": ["output"],
        },
        "tensor_shapes": {"inputs": input_shapes, "outputs": [output_shape]},
        "tensor_dtypes": {
            "inputs": ["torch.float16"] * len(input_shapes),
            "outputs": ["torch.float16"],
        },
        "tensor_types": {
            "inputs": ["input"] * len(input_shapes),
            "outputs": ["output"],
        },
    }


@pytest.mark.parametrize(
    ("target", "input_shapes", "output_shape", "equation"),
    [
        ("mm", [[2, 3], [3, 4]], [2, 4], "MK,KN->MN"),
        ("bmm", [[5, 2, 3], [5, 3, 4]], [5, 2, 4], "BMK,BKN->BMN"),
        ("linear", [[2, 3], [4, 3]], [2, 4], "B0K,NK->B0N"),
    ],
)
def test_exact_aten_contractions_gain_an_orojenesis_proof_view(
    target: str,
    input_shapes: list[list[int]],
    output_shape: list[int],
    equation: str,
) -> None:
    layer = _aten_layer(target, input_shapes, output_shape)

    proof = build_orojenesis_proof_layer(layer, analyzer=EinsumAnalyzer())

    assert proof is not None
    assert proof["semantic_op"]["kind"] == "einsum"
    assert proof["semantic_op"]["equation"] == equation
    assert proof["semantic_op"]["proof_source"] == {
        "kind": "aten",
        "target": target,
    }
    assert layer["semantic_op"]["kind"] == "aten"


def test_addmm_proof_excludes_the_non_contraction_bias_operand() -> None:
    layer = _aten_layer("addmm", [[4], [2, 3], [3, 4]], [2, 4])

    proof = build_orojenesis_proof_layer(layer, analyzer=EinsumAnalyzer())

    assert proof is not None
    assert proof["tensor_names"]["inputs"] == ["input_1", "input_2"]
    assert proof["tensor_shapes"]["inputs"] == [[2, 3], [3, 4]]
    assert proof["semantic_op"]["equation"] == "MK,KN->MN"


def _conv_arguments(
    *,
    stride: tuple[int, ...] = (1, 1),
    padding: tuple[int, ...] = (0, 0),
    groups: int = 1,
) -> list[object]:
    def literal(value: object) -> dict[str, object]:
        return {"value": value}

    return [
        {"tensor": 0},
        {"tensor": 1},
        literal(None),
        [literal(value) for value in stride],
        [literal(value) for value in padding],
        [literal(1) for _ in stride],
        literal(False),
        [literal(0) for _ in stride],
        literal(groups),
    ]


@pytest.mark.parametrize(
    ("input_shapes", "output_shape", "groups", "equation"),
    [
        (
            [[2, 3, 8, 8], [5, 3, 3, 3], [5]],
            [2, 5, 6, 6],
            1,
            "BC(P+R)(Q+S),OCRS->BOPQ",
        ),
        (
            [[2, 4, 8, 8], [4, 1, 3, 3], [4]],
            [2, 4, 6, 6],
            4,
            "BO(P+R)(Q+S),OCRS->BOPQ",
        ),
    ],
)
def test_reviewed_direct_convolutions_gain_exact_proof_views(
    input_shapes: list[list[int]],
    output_shape: list[int],
    groups: int,
    equation: str,
) -> None:
    layer = _aten_layer(
        "conv2d",
        input_shapes,
        output_shape,
        arguments=_conv_arguments(groups=groups),
    )

    proof = build_orojenesis_proof_layer(layer, analyzer=EinsumAnalyzer())

    assert proof is not None
    assert proof["tensor_shapes"]["inputs"] == input_shapes[:2]
    assert proof["semantic_op"]["equation"] == equation


def test_default_extended_native_conv_gains_the_same_exact_proof_view() -> None:
    layer = _aten_layer(
        "conv2d",
        [[2, 3, 8, 8], [5, 3, 3, 3], [5]],
        [2, 5, 6, 6],
    )
    layer["semantic_op"] = {
        "kind": "operation",
        "target": "conv2d",
        "operands": [{"tensor": 0}, {"tensor": 1}, {"tensor": 2}],
        "attributes": {},
        "effects": {
            "mutates": [],
            "aliases": [],
            "atomic": False,
            "opaque_library_call": True,
        },
    }

    proof = build_orojenesis_proof_layer(layer, analyzer=EinsumAnalyzer())

    assert proof is not None
    assert proof["tensor_shapes"]["inputs"] == [
        [2, 3, 8, 8],
        [5, 3, 3, 3],
    ]
    assert proof["semantic_op"]["equation"] == "BC(P+R)(Q+S),OCRS->BOPQ"
    assert proof["semantic_op"]["proof_source"] == {
        "kind": "extended_native",
        "target": "conv2d",
    }
    assert proof["semantic_op"]["effects"]["opaque_library_call"] is True


def test_nondefault_extended_native_conv_remains_fail_closed() -> None:
    layer = _aten_layer(
        "conv2d",
        [[2, 3, 8, 8], [5, 3, 3, 3]],
        [2, 5, 4, 4],
    )
    layer["semantic_op"] = {
        "kind": "operation",
        "target": "conv2d",
        "operands": [{"tensor": 0}, {"tensor": 1}],
        "attributes": {"stride": [2, 2]},
        "effects": {},
    }

    assert (
        build_orojenesis_proof_layer(layer, analyzer=EinsumAnalyzer()) is None
    )


@pytest.mark.parametrize(
    "arguments",
    [
        _conv_arguments(stride=(2, 2)),
        _conv_arguments(padding=(1, 1)),
        _conv_arguments(groups=2),
    ],
)
def test_unreviewed_convolutions_remain_fail_closed(
    arguments: list[object],
) -> None:
    layer = _aten_layer(
        "conv2d",
        [[1, 4, 8, 8], [6, 2, 3, 3]],
        [1, 6, 6, 6],
        arguments=arguments,
    )

    assert requires_tile_evidence(layer)
    assert (
        build_orojenesis_proof_layer(layer, analyzer=EinsumAnalyzer()) is None
    )


def test_unreviewed_depthwise_conv3d_remains_fail_closed() -> None:
    layer = _aten_layer(
        "conv3d",
        [[1, 4, 8, 8, 8], [4, 1, 3, 3, 3], [4]],
        [1, 4, 6, 6, 6],
        arguments=_conv_arguments(
            stride=(1, 1, 1),
            padding=(0, 0, 0),
            groups=4,
        ),
    )

    assert (
        build_orojenesis_proof_layer(layer, analyzer=EinsumAnalyzer()) is None
    )


@pytest.mark.parametrize("target", ["scaled_dot_product_attention"])
def test_unmodeled_contractions_are_never_treated_as_proven(
    target: str,
) -> None:
    layer = _aten_layer(target, [[1, 2, 3, 3], [2, 2, 1, 1]], [1, 2, 3, 3])

    assert requires_tile_evidence(layer)
    assert (
        build_orojenesis_proof_layer(layer, analyzer=EinsumAnalyzer()) is None
    )

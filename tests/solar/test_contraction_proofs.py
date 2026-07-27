from __future__ import annotations

import pytest

from solar.analysis.contraction_proofs import (
    build_orojenesis_proof_layer,
    requires_tile_evidence,
)
from solar.einsum.analyzer import EinsumAnalyzer


def _aten_layer(
    target: str,
    input_shapes: list[list[int]],
    output_shape: list[int],
) -> dict:
    return {
        "type": target,
        "semantic_op": {
            "kind": "aten",
            "target": target,
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


@pytest.mark.parametrize("target", ["conv2d", "scaled_dot_product_attention"])
def test_unmodeled_contractions_are_never_treated_as_proven(
    target: str,
) -> None:
    layer = _aten_layer(target, [[1, 2, 3, 3], [2, 2, 1, 1]], [1, 2, 3, 3])

    assert requires_tile_evidence(layer)
    assert (
        build_orojenesis_proof_layer(layer, analyzer=EinsumAnalyzer()) is None
    )

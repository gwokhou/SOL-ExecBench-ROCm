from __future__ import annotations

import pytest

from solar.analysis.operand_provenance import (
    contraction_external_source_dtypes,
    contraction_has_region_boundary_proof,
    contraction_operands_are_graph_external,
)


def _start(name: str, *, dtype: str | None = "torch.float16") -> dict:
    dtypes = [] if dtype is None else [dtype]
    return {
        "type": "start",
        "tensor_names": {"inputs": [], "outputs": [name]},
        "tensor_dtypes": {"inputs": [], "outputs": dtypes},
    }


def test_alias_cycle_missing_dtype_and_effects_fail_closed():
    layers = {
        "start": _start("input"),
        "missing_dtype": _start("untyped", dtype=None),
        "bad_alias": {
            "type": "view",
            "semantic_op": {
                "kind": "aten",
                "target": "view",
                "effects": {
                    "aliases": [
                        {"input": 9, "output": 0, "conditional": False},
                    ],
                },
            },
            "tensor_names": {"inputs": ["input"], "outputs": ["bad"]},
        },
        "mutating": {
            "type": "mul",
            "semantic_op": {
                "kind": "aten",
                "target": "mul",
                "effects": {"mutates": True},
            },
            "tensor_names": {"inputs": ["input"], "outputs": ["mutated"]},
        },
        "cycle": {
            "type": "view",
            "semantic_op": {
                "kind": "aten",
                "target": "view",
                "effects": {
                    "aliases": [
                        {"input": 0, "output": 0, "conditional": False},
                    ],
                },
            },
            "tensor_names": {"inputs": ["loop"], "outputs": ["loop"]},
        },
    }
    for tensor_name in ("untyped", "bad", "mutated", "loop"):
        contraction = {"tensor_names": {"inputs": [tensor_name]}}
        assert not contraction_operands_are_graph_external(contraction, layers)
        assert contraction_external_source_dtypes(contraction, layers) == set()


def test_conditional_alias_is_not_treated_as_zero_copy():
    layers = {
        "start": _start("input"),
        "conditional": {
            "type": "view",
            "semantic_op": {
                "kind": "aten",
                "target": "view",
                "effects": {
                    "aliases": [{"input": 0, "output": 0, "conditional": True}],
                },
            },
            "tensor_names": {"inputs": ["input"], "outputs": ["output"]},
        },
    }
    assert not contraction_operands_are_graph_external(
        {"tensor_names": {"inputs": ["output"]}},
        layers,
    )


def test_pointwise_add_operand_is_proven_tile_recomputable():
    layers = {
        "left": _start("left", dtype="torch.float32"),
        "right": _start("right", dtype="torch.float32"),
        "add": {
            "type": "add",
            "semantic_op": {
                "kind": "aten",
                "target": "add",
                "effects": {
                    "mutates": [],
                    "aliases": [],
                    "atomic": False,
                    "opaque_library_call": False,
                },
            },
            "tensor_names": {
                "inputs": ["left", "right"],
                "outputs": ["sum"],
            },
        },
    }
    contraction = {"tensor_names": {"inputs": ["left", "sum"]}}

    assert contraction_operands_are_graph_external(contraction, layers)
    assert contraction_external_source_dtypes(contraction, layers) == {
        "torch.float32",
    }


def _tensor_layer(
    target: str,
    input_name: str,
    output_name: str,
    *,
    shape: list[int] | None = None,
) -> dict:
    tensor_shape = shape or [2, 4]
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
            "inputs": [input_name],
            "outputs": [output_name],
        },
        "tensor_shapes": {
            "inputs": [tensor_shape],
            "outputs": [tensor_shape],
        },
        "tensor_dtypes": {
            "inputs": ["torch.float32"],
            "outputs": ["torch.float32"],
        },
    }


def test_region_boundary_proves_tile_local_scalar_output_chain():
    contraction = {
        **_tensor_layer("bmm", "left", "product"),
        "tensor_names": {
            "inputs": ["left", "right"],
            "outputs": ["product"],
        },
    }
    layers = {
        "bmm": contraction,
        "elu": _tensor_layer("elu", "product", "activated"),
        "div": _tensor_layer("div", "activated", "scaled"),
    }
    region = {
        "layers": ["bmm", "elu", "div"],
        "external_inputs": ["left", "right"],
        "external_outputs": ["scaled"],
    }

    assert contraction_has_region_boundary_proof(contraction, region, layers)


@pytest.mark.parametrize(
    "failure", ["tensor_operand", "shape", "fanout", "normalization"]
)
def test_region_boundary_output_proof_fails_closed(failure: str):
    contraction = {
        **_tensor_layer("bmm", "left", "product"),
        "tensor_names": {
            "inputs": ["left", "right"],
            "outputs": ["product"],
        },
    }
    postprocess = _tensor_layer("elu", "product", "result")
    layers = {"bmm": contraction, "postprocess": postprocess}
    region = {
        "layers": ["bmm", "postprocess"],
        "external_inputs": ["left", "right"],
        "external_outputs": ["result"],
    }
    if failure == "tensor_operand":
        postprocess["tensor_names"]["inputs"].append("other")
        postprocess["tensor_shapes"]["inputs"].append([2, 4])
        postprocess["tensor_dtypes"]["inputs"].append("torch.float32")
    elif failure == "shape":
        postprocess["tensor_shapes"]["outputs"] = [[4, 2]]
    elif failure == "fanout":
        layers["second_consumer"] = _tensor_layer("relu", "product", "second")
        region["layers"].append("second_consumer")
    else:
        postprocess["semantic_op"]["target"] = "softmax"

    assert not contraction_has_region_boundary_proof(
        contraction, region, layers
    )

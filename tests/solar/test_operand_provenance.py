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


def _broadcast_region(shape, side_shape, *, target="add", tracked_first=True):
    contraction = _tensor_layer("bmm", "left", "product", shape=shape)
    contraction["tensor_names"]["inputs"] = ["left", "right"]
    post = _tensor_layer(target, "product", "shifted", shape=shape)
    post["tensor_names"]["inputs"] = ["product", "side"]
    post["tensor_shapes"]["inputs"] = [shape, side_shape]
    post["tensor_dtypes"]["inputs"] *= 2
    if not tracked_first:
        for field in ("tensor_names", "tensor_shapes", "tensor_dtypes"):
            post[field]["inputs"].reverse()
    layers = {
        "bmm": contraction,
        "binary": post,
        "relu": _tensor_layer("relu", "shifted", "result", shape=shape),
    }
    region = {
        "layers": list(layers),
        "external_inputs": ["left", "right", "side"],
        "external_outputs": ["result"],
    }
    return contraction, region, layers


@pytest.mark.parametrize(
    ("shape", "side_shape"),
    [
        ([2, 4], [4]),
        ([2, 4], [2, 1]),
        ([2, 4], []),
        ([2, 4], [2, 4]),
        ([2, 3, 4], [3, 1]),
        ([2, 3, 4], [2, 1, 4]),
    ],
    ids=[
        "last-axis",
        "first-axis",
        "scalar",
        "same-shape",
        "middle-axis",
        "outer-axes",
    ],
)
def test_region_boundary_proves_static_broadcast(shape, side_shape):
    assert contraction_has_region_boundary_proof(
        *_broadcast_region(shape, side_shape)
    )


@pytest.mark.parametrize(
    "target", ["add", "sub", "mul", "div", "maximum", "minimum"]
)
@pytest.mark.parametrize("tracked_first", [True, False], ids=["left", "right"])
def test_region_boundary_proves_binary_operand_order(target, tracked_first):
    assert contraction_has_region_boundary_proof(
        *_broadcast_region(
            [2, 4], [4], target=target, tracked_first=tracked_first
        )
    )


@pytest.mark.parametrize(
    "failure",
    [
        "incompatible",
        "rank-expansion",
        "dynamic-side",
        "dynamic-tracked",
        "zero",
        "negative",
        "bool",
        "float",
        "not-boundary",
        "internal-side",
        "missing-shape",
        "extra-shape",
        "missing-dtype",
        "promotion",
        "output-shape",
        "output-dtype",
        "extra-output",
        "alias",
        "mutation",
        "atomic",
        "opaque",
        "duplicate-tracked",
        "three-inputs",
        "reduction",
        "fanout",
    ],
)
def test_region_boundary_rejects_unproven_broadcast(failure):
    contraction, region, layers = _broadcast_region([2, 4], [4])
    post = layers["binary"]
    shapes = post["tensor_shapes"]
    invalid_shapes = {
        "incompatible": [2],
        "rank-expansion": [1, 2, 4],
        "dynamic-side": ["N"],
        "zero": [0],
        "negative": [-1],
        "bool": [True],
        "float": [4.0],
    }
    if failure in invalid_shapes:
        shapes["inputs"][1] = invalid_shapes[failure]
    elif failure == "dynamic-tracked":
        contraction["tensor_shapes"]["outputs"][0][0] = "N"
    elif failure == "not-boundary":
        region["external_inputs"].remove("side")
    elif failure == "internal-side":
        layers["producer"] = _tensor_layer("relu", "left", "side")
        region["layers"].append("producer")
    elif failure == "missing-shape":
        shapes["inputs"].pop()
    elif failure == "extra-shape":
        shapes["inputs"].append([4])
    elif failure == "missing-dtype":
        post["tensor_dtypes"]["inputs"].pop()
    elif failure == "promotion":
        post["tensor_dtypes"]["inputs"][1] = "torch.float16"
    elif failure == "output-shape":
        shapes["outputs"] = [[4, 2]]
    elif failure == "output-dtype":
        post["tensor_dtypes"]["outputs"] = ["torch.float16"]
    elif failure == "extra-output":
        post["tensor_names"]["outputs"].append("extra")
    elif failure in ("alias", "mutation", "atomic", "opaque"):
        key = {
            "alias": "aliases",
            "mutation": "mutates",
            "opaque": "opaque_library_call",
        }.get(failure, failure)
        post["semantic_op"]["effects"][key] = True
    elif failure == "duplicate-tracked":
        post["tensor_names"]["inputs"][1] = "product"
    elif failure == "three-inputs":
        post["tensor_names"]["inputs"].append("other")
        shapes["inputs"].append([4])
        post["tensor_dtypes"]["inputs"].append("torch.float32")
        region["external_inputs"].append("other")
    elif failure == "reduction":
        post["semantic_op"]["target"] = "max"
    else:
        layers["fanout"] = _tensor_layer("relu", "product", "extra")
        region["layers"].append("fanout")
    assert not contraction_has_region_boundary_proof(
        contraction, region, layers
    )


def test_region_boundary_proves_multiple_external_inputs_along_chain():
    contraction, region, layers = _broadcast_region([2, 4], [4])
    second = _tensor_layer("mul", "result", "scaled")
    second["tensor_names"]["inputs"].append("scale")
    second["tensor_shapes"]["inputs"].append([2, 1])
    second["tensor_dtypes"]["inputs"].append("torch.float32")
    layers["scale"] = second
    region["layers"].append("scale")
    region["external_inputs"].append("scale")
    region["external_outputs"] = ["scaled"]
    assert contraction_has_region_boundary_proof(contraction, region, layers)
    region["external_inputs"].remove("scale")
    assert not contraction_has_region_boundary_proof(
        contraction, region, layers
    )

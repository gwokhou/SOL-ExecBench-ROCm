from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from solar.einsum import pytorch_to_einsum
from solar.einsum.pytorch_to_einsum import PyTorchToEinsum


def _tensor(
    shape: list[int],
    dtype: str,
    *,
    inputs: list[str] | None = None,
    outputs: list[str] | None = None,
    node_type: str = "hidden-tensor",
) -> dict:
    return {
        "type": node_type,
        "output_shapes": [shape],
        "output_dtypes": [dtype],
        "connections": {"inputs": inputs or [], "outputs": outputs or []},
    }


def _operation(
    node_type: str,
    *,
    inputs: list[str],
    outputs: list[str],
    input_shapes: list[list[int]],
    input_dtypes: list[str],
    output_shapes: list[list[int]] | None = None,
    output_dtypes: list[str] | None = None,
    raw_attributes: str = "",
) -> dict:
    return {
        "type": node_type,
        "input_shapes": input_shapes,
        "output_shapes": output_shapes or ([input_shapes[0]] if input_shapes else []),
        "input_dtypes": input_dtypes,
        "output_dtypes": output_dtypes or ["torch.float32"],
        "input_types": ["input"] * len(input_shapes),
        "module_args": {"raw_attributes": raw_attributes},
        "connections": {"inputs": inputs, "outputs": outputs},
    }


def test_repair_recovers_parameters_and_external_tensor_arguments():
    converter = PyTorchToEinsum()
    linear = _operation(
        "linear",
        inputs=["x"],
        outputs=["linear_out"],
        input_shapes=[[2, 3]],
        input_dtypes=["torch.float16"],
        output_shapes=[[2, 4]],
        raw_attributes=(
            "[[Tensor(shape=(2, 3), dtype=torch.float16), "
            "Tensor(shape=(4, 3), dtype=torch.float16), "
            "Tensor(shape=(4,), dtype=torch.float16)], {}]"
        ),
    )
    where = _operation(
        "where",
        inputs=["condition", "left"],
        outputs=["where_out"],
        input_shapes=[[2, 2], [2, 2]],
        input_dtypes=["torch.bool", "torch.float16"],
        raw_attributes=(
            "[[Tensor(shape=(2, 2), dtype=torch.bool), "
            "Tensor(shape=(2, 2), dtype=torch.float16), "
            "Tensor(shape=(2, 2), dtype=torch.float16)], {}]"
        ),
    )
    layers = {
        "x": _tensor([2, 3], "torch.float16", outputs=["linear"]),
        "linear": linear,
        "linear_out": _tensor([2, 4], "torch.float32", inputs=["linear"]),
        "condition": _tensor([2, 2], "torch.bool", outputs=["where"]),
        "left": _tensor([2, 2], "torch.float16", outputs=["where"]),
        "where": where,
        "where_out": _tensor([2, 2], "torch.float32", inputs=["where"]),
    }
    converter._repair_torchview_quirks(
        layers,
        ["linear", "where"],
        ["x", "linear_out", "condition", "left", "where_out"],
    )
    assert linear["input_shapes"] == [[2, 3], [4, 3], [4]]
    assert linear["input_types"] == ["input", "weight", "weight"]
    assert linear["connections"]["inputs"][-2:] == [
        "linear.auxiliary-tensor_1",
        "linear.auxiliary-tensor_2",
    ]
    assert where["connections"]["inputs"][-1] == "where.auxiliary-tensor_2"
    assert (
        layers["where.auxiliary-tensor_2"]["module_args"]["recovered_from"]
        == "exact_call_signature"
    )
    assert where["output_dtypes"] == ["torch.float16"]


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (
            "[[Tensor(shape=(2, 3), dtype=torch.float16), "
            "Tensor(shape=(unknown,), dtype=torch.float16)], {}]",
            "cannot recover exact parameter metadata",
        ),
        (
            "[[Tensor(shape=(2, 3), dtype=torch.float16), "
            "Tensor(shape=(4, 3), dtype=torch.float16), "
            "Tensor(shape=(4,), dtype=torch.float16)], {}]",
            "parameter tensor order is incomplete",
        ),
    ],
)
def test_repair_rejects_incomplete_parameter_metadata(raw, message):
    converter = PyTorchToEinsum()
    node = _operation(
        "linear",
        inputs=[],
        outputs=[],
        input_shapes=[[2, 3]],
        input_dtypes=["torch.float16"],
        raw_attributes=raw,
    )
    if "unknown" not in raw:
        # Index two is recoverable, but index one is absent: this is an unsafe gap.
        converter._PARAMETER_TENSOR_INDICES = {
            **converter._PARAMETER_TENSOR_INDICES,
            "linear": {2},
        }
    with pytest.raises(pytorch_to_einsum.ConversionError, match=message):
        converter._repair_torchview_quirks({"linear": node}, ["linear"], [])


def test_repair_reconnects_dropped_and_split_tensor_edges():
    converter = PyTorchToEinsum()
    norm = _operation(
        "norm",
        inputs=["x"],
        outputs=["dangling_scalar"],
        input_shapes=[[2, 2]],
        input_dtypes=["torch.float16"],
        output_shapes=[[]],
        output_dtypes=["torch.float16"],
    )
    divide = _operation(
        "div",
        inputs=["x"],
        outputs=["divide_out"],
        input_shapes=[[2, 2]],
        input_dtypes=["torch.float16"],
        raw_attributes=(
            "[[Tensor(shape=(2, 2), dtype=torch.float16), "
            "Tensor(shape=(), dtype=torch.float16)], {}]"
        ),
    )
    producer = _operation(
        "add",
        inputs=["x"],
        outputs=["dead_end"],
        input_shapes=[[2, 2]],
        input_dtypes=["torch.float16"],
        output_shapes=[[2]],
        output_dtypes=["torch.float16"],
    )
    consumer = _operation(
        "relu",
        inputs=["orphan"],
        outputs=["consumer_out"],
        input_shapes=[[2]],
        input_dtypes=["torch.float16"],
    )
    layers = {
        "x": _tensor([2, 2], "torch.float16", outputs=["norm", "div", "producer"]),
        "norm": norm,
        "dangling_scalar": _tensor([], "torch.float16", inputs=["norm"]),
        "div": divide,
        "divide_out": _tensor([2, 2], "torch.float32", inputs=["div"]),
        "producer": producer,
        "dead_end": _tensor([2], "torch.float16", inputs=["producer"]),
        "orphan": _tensor([2], "torch.float16", outputs=["consumer"]),
        "consumer": consumer,
        "consumer_out": _tensor([2], "torch.float32", inputs=["consumer"]),
    }
    converter._repair_torchview_quirks(
        layers,
        ["norm", "div", "producer", "consumer"],
        ["x", "dangling_scalar", "divide_out", "dead_end", "orphan", "consumer_out"],
    )
    assert divide["input_shapes"] == [[2, 2], []]
    assert "div" in layers["dangling_scalar"]["connections"]["outputs"]
    assert layers["orphan"]["connections"]["inputs"] == ["producer"]
    assert converter._tensor_to_producer_op["orphan"] == "producer"


@pytest.mark.parametrize(
    ("node_type", "input_dtypes", "module_args", "expected"),
    [
        ("half", ["torch.float32"], {}, "torch.float16"),
        ("long", ["torch.float16"], {}, "torch.int64"),
        ("eq", ["torch.float16", "torch.float16"], {}, "torch.bool"),
        ("bitwise_and", ["torch.int32"], {}, "torch.int32"),
        (
            "to",
            ["torch.float16"],
            {"call_arguments": [{"dtype": "bfloat16"}]},
            "torch.bfloat16",
        ),
        ("view", ["torch.float16"], {}, "torch.float16"),
        ("where", ["torch.bool", "torch.bfloat16"], {}, "torch.bfloat16"),
        ("add", ["torch.float16", "torch.float32"], {}, "torch.float32"),
    ],
)
def test_repair_uses_operation_specific_output_dtype(
    node_type, input_dtypes, module_args, expected
):
    converter = PyTorchToEinsum()
    inputs = [f"input_{index}" for index in range(len(input_dtypes))]
    layers = {
        tensor_id: _tensor([2], dtype, outputs=["op"])
        for tensor_id, dtype in zip(inputs, input_dtypes)
    }
    layers["op"] = _operation(
        node_type,
        inputs=inputs,
        outputs=["out"],
        input_shapes=[[2]] * len(inputs),
        input_dtypes=list(input_dtypes),
    )
    layers["op"]["module_args"] = module_args
    layers["out"] = _tensor([2], "torch.float32", inputs=["op"])
    converter._repair_torchview_quirks(layers, ["op"], [*inputs, "out"])
    assert layers["op"]["output_dtypes"] == [expected]
    assert layers["out"]["output_dtypes"] == [expected]


def test_tensor_shape_consistency_reports_cross_layer_conflicts():
    converter = PyTorchToEinsum()
    valid: dict[str, Any] = {
        "layers": {
            "a": {
                "tensor_names": {"inputs": [], "outputs": ["shared"]},
                "tensor_shapes": {"inputs": [], "outputs": [[2, 3]]},
            },
            "b": {
                "tensor_names": {"inputs": ["shared"], "outputs": ["out"]},
                "tensor_shapes": {"inputs": [[2, 3]], "outputs": [[2, 3]]},
            },
        }
    }
    converter._validate_tensor_shape_consistency(valid)
    invalid = deepcopy(valid)
    invalid["layers"]["b"]["tensor_shapes"]["inputs"] = [[3, 2]]
    with pytest.raises(ValueError, match="inconsistency"):
        converter._validate_tensor_shape_consistency(invalid)

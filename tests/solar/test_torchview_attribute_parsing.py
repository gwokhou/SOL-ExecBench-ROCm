from __future__ import annotations

from solar.graph.torchview.attribute_parsing import (
    parse_operation_attributes,
)
from solar.graph.torchview.processor import TorchviewProcessor


def _tensor() -> dict[str, bool]:
    return {"tensor_placeholder": True}


def test_semantic_argument_encoding_preserves_tensor_order_recursively() -> (
    None
):
    parsed = parse_operation_attributes(
        "identity",
        [_tensor(), {"nested": [_tensor(), 3]}],
        {"dtype": "__torch_float16__"},
    )

    assert parsed["call_arguments"] == [
        {"tensor": 0},
        {"nested": [{"tensor": 1}, {"value": 3}]},
    ]
    assert parsed["call_kwargs"] == {"dtype": {"dtype": "float16"}}


def test_transpose_kwargs_override_positional_diagnostics() -> None:
    parsed = parse_operation_attributes(
        "transpose",
        [_tensor(), 0, 1],
        {"dim0": 2, "dim1": 3},
    )

    assert parsed["dim0"] == 2
    assert parsed["dim1"] == 3
    assert parsed["transpose_dims"] == [2, 3]


def test_shape_and_reduction_rules_normalize_sequences() -> None:
    reshape = parse_operation_attributes(
        "reshape",
        [_tensor(), (2, 3, 4)],
        {},
    )
    reduction = parse_operation_attributes(
        "sum",
        [_tensor()],
        {"dim": (1, 2), "keepdim": True},
    )

    assert reshape["target_shape"] == [2, 3, 4]
    assert reduction["dim"] == [1, 2]
    assert reduction["keepdim"] is True


def test_converter_attribute_boundary_uses_typed_parser() -> None:
    attributes = "[[Tensor(shape=(2, 3), dtype=torch.float32), 1, 0], {}]"

    parsed = TorchviewProcessor()._parse_torchview_attributes(
        attributes,
        "transpose",
    )

    assert parsed["call_arguments"] == [
        {"tensor": 0},
        {"value": 1},
        {"value": 0},
    ]
    assert parsed["transpose_dims"] == [1, 0]

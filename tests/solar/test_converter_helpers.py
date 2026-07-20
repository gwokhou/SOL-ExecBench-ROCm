from __future__ import annotations

import networkx as nx
import pytest

from solar.einsum import pytorch_to_einsum
from solar.einsum.pytorch_to_einsum import PyTorchToEinsum


def test_parsing_and_dtype_helpers_cover_recorded_call_variants():
    converter = PyTorchToEinsum(debug=True)
    assert converter.debug
    assert converter.einsum_analyzer is not None
    assert pytorch_to_einsum._product([]) == 1
    assert pytorch_to_einsum._product([2, 3, 4]) == 24

    assert converter._parse_einsum_from_raw_attributes({}) is None
    assert (
        converter._parse_einsum_from_raw_attributes(
            {"raw_attributes": "[['bij,bjk->bik', Tensor(...)], {}]"}
        )
        == "BIJ,BJK->BIK"
    )
    assert (
        converter._parse_einsum_from_raw_attributes(
            {"raw_attributes": "no equation here"}
        )
        is None
    )
    assert converter._convert_einsum_to_solar_format("") == ""
    assert converter._convert_einsum_to_solar_format("ij") == "ij"
    assert converter._convert_einsum_to_solar_format("i->i->i") == "I->I->I"

    assert converter._parse_reduction_args_from_raw_attributes(
        {"dim": 1, "keepdim": True}
    ) == ([1], True)
    assert converter._parse_reduction_args_from_raw_attributes({"dim": (0, 2)}) == (
        [0, 2],
        False,
    )
    assert converter._parse_reduction_args_from_raw_attributes({}) == (None, False)
    assert converter._parse_reduction_args_from_raw_attributes(
        {"raw_attributes": "{dim: [1, -2], keepdim: True}"}
    ) == ([1, -2], True)
    assert converter._parse_reduction_args_from_raw_attributes(
        {"raw_attributes": "{dim: -1, keepdim: False}"}
    ) == ([-1], False)

    raw = {
        "raw_attributes": (
            "Tensor(shape=(), dtype=torch.float32), "
            "Tensor(shape=(2, 3,), dtype=torch.int64), "
            "Tensor(shape=(bad,), dtype=torch.float16)"
        )
    }
    assert converter._tensor_arg_shapes_from_raw({}) == []
    assert converter._tensor_arg_shapes_from_raw(raw) == [(), (2, 3), None]
    assert converter._tensor_arg_dtypes_from_raw(raw) == [
        "torch.float32",
        "torch.int64",
        "torch.float16",
    ]
    assert converter._bits_of_dtype(None) == 32
    assert converter._bits_of_dtype("torch.float16") == 16
    assert converter._bits_of_dtype("unknown") == 32


@pytest.mark.parametrize(
    ("node", "expected"),
    [
        ({"type": "relu"}, False),
        (
            {
                "type": "linear",
                "input_shapes": [[2, 3], [4, 3], [4]],
                "input_types": ["input", "weight", "weight"],
            },
            True,
        ),
        (
            {
                "type": "linear",
                "input_shapes": [[2, 3], [4, 3], [4]],
                "input_types": [],
            },
            True,
        ),
        ({"type": "linear", "module_args": {"bias": True}}, True),
        ({"type": "linear", "notes": "contains Bias tensor"}, True),
        (
            {
                "type": "linear",
                "input_shapes": [[2, 3], [4, 3]],
                "input_types": ["input", "weight"],
            },
            False,
        ),
    ],
)
def test_linear_bias_detection_variants(node, expected):
    assert PyTorchToEinsum()._should_split_linear_with_bias(node) is expected


def test_name_shape_alignment_grows_and_trims_each_side():
    converter = PyTorchToEinsum()
    names, shapes = converter._align_tensor_names_and_shapes(
        {"inputs": ["one"], "outputs": ["out", "extra"]},
        {"inputs": [[1], [2]], "outputs": [[3]]},
        {"id": "node"},
    )
    assert names == {"inputs": ["one", "node.Input_1"], "outputs": ["out"]}
    assert shapes == {"inputs": [[1], [2]], "outputs": [[3]]}

    names, _ = converter._align_tensor_names_and_shapes(
        {"inputs": ["one", "extra"], "outputs": []},
        {"inputs": [[1]], "outputs": [[2], [3]]},
        {},
    )
    assert names == {
        "inputs": ["one"],
        "outputs": ["unknown.Output_0", "unknown.Output_1"],
    }


def test_deferred_input_resolution_has_three_explicit_outcomes():
    connections: list[str | None] = [None]
    PyTorchToEinsum._fill_deferred_input_connections(
        "op", connections, [0], ["tensor"], ["producer"], {}
    )
    assert connections == ["producer"]

    connections = [None]
    PyTorchToEinsum._fill_deferred_input_connections(
        "op", connections, [0], ["tensor"], [], {"tensor": "start"}
    )
    assert connections == ["start"]

    with pytest.raises(ValueError, match="cannot uniquely resolve"):
        PyTorchToEinsum._fill_deferred_input_connections(
            "op", [None], [0], ["tensor"], ["a", "b"], {}
        )


def test_input_resolution_handles_weights_starts_ops_and_repaired_tensors():
    converter = PyTorchToEinsum()
    converter._tensor_to_producer_op = {"hidden": "producer"}
    graph = nx.DiGraph()
    graph.add_edges_from([("start", "op"), ("producer", "op"), ("direct", "op")])
    node = {"input_types": ["weight", "input", "input", "input"]}
    assert converter._resolve_input_connections(
        "op",
        node,
        graph,
        ["weight", "original_start", "direct", "hidden"],
        {"original_start": "start"},
    ) == ["weight", "start", "direct", "producer"]


@pytest.mark.parametrize(
    ("node_type", "elementwise", "reduction", "real"),
    [
        ("add", "add", "none", False),
        ("sum", "copy", "add", False),
        ("prod", "copy", "mul", False),
        ("max", "copy", "max", False),
        ("mystery", "mul", "add", True),
    ],
)
def test_non_strict_failed_representation_is_semantically_classified(
    node_type, elementwise, reduction, real
):
    result = PyTorchToEinsum()._failed_operation_representation(
        "node", node_type, RuntimeError("not handled")
    )
    assert (result.elementwise_op, result.reduction_op, result.is_real_einsum) == (
        elementwise,
        reduction,
        real,
    )


def test_strict_failed_representation_distinguishes_aten_and_unknown():
    converter = PyTorchToEinsum(strict=True)
    aten = converter._failed_operation_representation("node", "add", ValueError())
    assert aten.is_einsum_supportable
    assert aten.equation == ""
    with pytest.raises(pytorch_to_einsum.ConversionError, match="cannot exactly"):
        converter._failed_operation_representation(
            "node", "not_supported", ValueError()
        )


def test_supportability_recognizes_aliases_prefixes_and_explicit_rejections():
    converter = PyTorchToEinsum()
    assert converter._is_operation_supportable("add")
    assert converter._is_operation_supportable("aten.add")
    assert converter._is_operation_supportable("torch.relu")
    assert not converter._is_operation_supportable("if")
    assert converter._is_operation_supportable("project_specific_op")

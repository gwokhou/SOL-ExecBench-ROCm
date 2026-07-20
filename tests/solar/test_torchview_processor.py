from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from torch import nn

from solar.common.types import NodeInfo
from solar.graph.torchview_processor import TorchviewProcessor


def _node(class_name: str, **attributes):
    instance = type(class_name, (), {})()
    for name, value in attributes.items():
        setattr(instance, name, value)
    return instance


def _tensor_node(
    name: str,
    node_id: str,
    shape: tuple[int, ...],
    dtype: torch.dtype = torch.float16,
):
    return _node(
        "TensorNode",
        name=name,
        operation=name,
        node_id=node_id,
        tensor_shape=shape,
        tensor_dtype=dtype,
        parents=[],
    )


def test_process_edge_list_preserves_argument_order_types_and_yaml(tmp_path: Path):
    source = _tensor_node("input-tensor", "input", (2, 3))
    weight = _tensor_node("parameter-tensor", "weight", (4, 3))
    output = _tensor_node("output-tensor", "output", (2, 4))
    function = _node(
        "FunctionNode",
        name="linear",
        operation="linear",
        node_id="linear",
        inputs=[source, weight],
        outputs=[output],
        parents=[],
        ordered_input_nodes=[weight, source],
        attributes=(
            "[[Tensor(shape=(2, 3), dtype=torch.float16), "
            "Tensor(shape=(4, 3), dtype=torch.float16)], {bias: False}]"
        ),
        kwargs={"alpha": 1, "ignored": torch.ones(1)},
    )
    graph = SimpleNamespace(
        edge_list=[(source, function), (weight, function), (function, output)]
    )
    processor = TorchviewProcessor(debug=True)
    nodes = processor.process_graph(graph, str(tmp_path), "fake-kernel")
    by_type = {node.type: node for node in nodes}
    assert by_type["linear"].input_nodes == [
        by_type["parameter-tensor"].node_id,
        by_type["input-tensor"].node_id,
    ]
    assert by_type["linear"].input_types == ["weight", "input"]
    assert by_type["linear"].output_shapes == [[2, 4]]
    assert by_type["linear"].module_args["call_kwargs"]["bias"] == {"value": False}
    assert by_type["linear"].module_args["alpha"] == 1
    document = yaml.safe_load((tmp_path / "pytorch_graph.yaml").read_text())
    assert document["model_name"] == "fake-kernel"
    assert set(document["layers"]) == {node.node_id for node in nodes}


@pytest.mark.parametrize("edge", [(object(),), (object(), object(), object())])
def test_edge_list_requires_exactly_two_nodes(edge):
    with pytest.raises(ValueError, match="Expected"):
        TorchviewProcessor()._extract_from_edge_list(SimpleNamespace(edge_list=[edge]))


def test_edge_list_rejects_unknown_nodes_and_parameter_outputs():
    with pytest.raises(TypeError, match="Invalid node type"):
        TorchviewProcessor()._extract_from_edge_list(
            SimpleNamespace(edge_list=[(object(), object())])
        )

    function = _node(
        "FunctionNode",
        name="bad",
        node_id="bad",
        inputs=[],
        outputs=[],
        parents=[],
    )
    parameter = _tensor_node("parameter-tensor", "parameter", (2,))
    processor = TorchviewProcessor()
    processor._reset_state()
    with pytest.raises(AssertionError, match="should only appear as input"):
        processor._extract_from_edge_list(
            SimpleNamespace(edge_list=[(function, parameter)])
        )


@pytest.mark.parametrize(
    ("tensor_type", "expected_inputs", "expected_outputs"),
    [
        ("input-tensor", [], [[2, 3]]),
        ("auxiliary-tensor", [], [[2, 3]]),
        ("parameter-tensor", [], [[2, 3]]),
        ("output-tensor", [[2, 3]], []),
        ("hidden-tensor", [[2, 3]], [[2, 3]]),
        ("unclassified", [[2, 3]], [[2, 3]]),
    ],
)
def test_tensor_node_shape_and_dtype_classification(
    tensor_type, expected_inputs, expected_outputs
):
    processor = TorchviewProcessor()
    node = _tensor_node(tensor_type, tensor_type, (2, 3), torch.int64)
    assert processor._extract_shapes(node, tensor_type) == (
        expected_inputs,
        expected_outputs,
    )
    input_dtypes, output_dtypes = processor._extract_dtypes(node, tensor_type)
    if expected_inputs:
        assert input_dtypes == ["torch.int64"]
    else:
        assert input_dtypes == []
    if expected_outputs:
        assert output_dtypes == ["torch.int64"]
    else:
        assert output_dtypes == []


def test_generic_shape_and_dtype_fallbacks():
    processor = TorchviewProcessor()
    input_value = SimpleNamespace(shape=(2,), dtype="custom.float")
    output_value = SimpleNamespace(tensor_shape=(3,), tensor_dtype=torch.float32)
    node = _node(
        "FunctionNode",
        name="custom",
        inputs=[input_value],
        outputs=[output_value],
        parents=[],
    )
    assert processor._extract_shapes(node, "custom") == ([[2]], [[3]])
    assert processor._extract_dtypes(node, "custom") == (
        ["custom.float"],
        ["torch.float32"],
    )

    fallback = _node(
        "FunctionNode",
        name="fallback",
        input_shape=((2, 3),),
        output_shape=((2, 4),),
        input_dtype=torch.float16,
        output_dtype=(torch.float16,),
        parents=[],
    )
    shapes = processor._extract_shapes(fallback, "fallback")
    assert shapes == ([[2, 3]], [[2, 4]])
    assert processor._extract_dtypes(fallback, "fallback", None, *shapes) == (
        ["torch.float16"],
        ["torch.float16"],
    )

    model = nn.Linear(3, 4, dtype=torch.bfloat16)
    missing = _node("FunctionNode", name="missing", parents=[])
    assert processor._extract_dtypes(missing, "missing", model, [[2, 3]], [[2, 4]]) == (
        ["torch.bfloat16"],
        ["torch.bfloat16"],
    )


def test_module_extraction_prefers_live_module_then_attribute_fallback():
    processor = TorchviewProcessor()
    live = _node("ModuleNode", name="linear", module=nn.Linear(3, 4, bias=False))
    info = processor._extract_module_info(live)["module_args"]
    assert info["module_type"] == "Linear"
    assert info["in_features"] == 3
    assert info["out_features"] == 4
    assert info["bias"] is False

    fallback = _node(
        "ModuleNode",
        name="conv",
        attributes=(
            "Conv2d(training=False, in_channels=3, kernel_size=(3, 3), "
            "padding=None, dtype=torch.float16, scale=1.5)"
        ),
    )
    parsed = processor._extract_module_info(fallback)["module_args"]
    assert parsed == {
        "module_type": "Conv2d",
        "training": False,
        "in_channels": 3,
        "kernel_size": (3, 3),
        "padding": None,
        "dtype": "torch.float16",
        "scale": 1.5,
    }
    assert processor._parse_module_attributes_string("") == {}
    assert processor._parse_module_attributes_string("not a module") == {}


@pytest.mark.parametrize(
    ("node_name", "attributes", "expected"),
    [
        (
            "transpose",
            "[[Tensor(shape=(2, 3), dtype=torch.float32), 0, 1], {}]",
            {"transpose_dims": [0, 1]},
        ),
        (
            "transpose",
            "[[Tensor(shape=(2, 3), dtype=torch.float32)], {dim0: 1, dim1: 0}]",
            {"transpose_dims": [1, 0]},
        ),
        (
            "permute",
            "[[Tensor(shape=(2, 3), dtype=torch.float32), [1, 0]], {}]",
            {"permute_dims": [1, 0]},
        ),
        (
            "permute",
            "[[Tensor(shape=(2, 3), dtype=torch.float32)], {dims: [1, 0]}]",
            {"permute_dims": [1, 0]},
        ),
        (
            "t",
            "[[Tensor(shape=(2, 3), dtype=torch.float32)], {}]",
            {"transpose_dims": [1, 0]},
        ),
        (
            "view",
            "[[Tensor(shape=(2, 3), dtype=torch.float32), 6, 1], {}]",
            {"target_shape": [6, 1]},
        ),
        (
            "reshape",
            "[[Tensor(shape=(2, 3), dtype=torch.float32), [3, 2]], {}]",
            {"target_shape": [3, 2]},
        ),
        (
            "sum",
            "[[Tensor(shape=(2, 3), dtype=torch.float32)], {dim: -1, keepdim: True}]",
            {"dim": [-1], "keepdim": True},
        ),
        (
            "mean",
            "[[Tensor(shape=(2, 3), dtype=torch.float32), [0, 1]], {}]",
            {"dim": [0, 1]},
        ),
    ],
)
def test_torchview_attribute_parser_recovers_operation_arguments(
    node_name, attributes, expected
):
    parsed = TorchviewProcessor()._parse_torchview_attributes(attributes, node_name)
    for key, value in expected.items():
        assert parsed[key] == value
    assert parsed["call_arguments"][0] == {"tensor": 0}


def test_attribute_eval_preserves_nested_values_dtype_slice_and_ellipsis():
    processor = TorchviewProcessor()
    attributes = (
        "[[Tensor(shape=(2, 3), dtype=torch.float32), slice(1, None, 2), "
        "Ellipsis, torch.float16], {options: {flag: True}}]"
    )
    args, kwargs = processor._eval_attributes_string(attributes)
    assert args == [
        {"tensor_placeholder": True},
        {"slice": [1, None, 2]},
        "__ellipsis__",
        "__torch_float16__",
    ]
    assert kwargs == {"options": {"flag": True}}
    parsed = processor._parse_torchview_attributes(attributes, "getitem")
    assert parsed["call_arguments"][1] == {
        "slice": [{"value": 1}, {"value": None}, {"value": 2}]
    }
    assert parsed["call_arguments"][2] == {"value": "__ellipsis__"}
    assert parsed["call_arguments"][3] == {"dtype": "float16"}
    assert processor._eval_attributes_string("invalid(") == (None, None)
    assert processor._parse_torchview_attributes("", "add") == {}


def test_function_info_adds_scalar_kwargs_but_not_tensor_kwargs():
    node = _node(
        "FunctionNode",
        name="add",
        attributes="[[Tensor(shape=(2,), dtype=torch.float32), 2], {}]",
        kwargs={"alpha": 2, "other": torch.ones(1)},
    )
    info = TorchviewProcessor()._extract_module_info(node)["module_args"]
    assert info["function_name"] == "add"
    assert info["alpha"] == 2
    assert "other" not in info


def test_hierarchical_names_handle_duplicates_cycles_and_tensor_bridges():
    processor = TorchviewProcessor()
    processor._reset_state()
    root = _node("ModuleNode", name="Block", parents=[])
    nested = _node("ModuleNode", name="Block", parents=[root])
    tensor = _node("TensorNode", name="hidden-tensor", parents=[nested])
    function = _node("FunctionNode", name="relu", parents=[tensor])
    hierarchy = processor._get_module_hierarchy_with_ids(function)
    assert [name for name, _ in hierarchy] == ["Block", "Block"]
    processor._prescan_module_hierarchy({"fn": function}, ["fn"])
    assert processor._index_hierarchy(hierarchy) == ["Block_0", "Block_1"]
    assert processor._generate_hierarchical_name(function).startswith(
        "Model.Block_0.Block_1.relu"
    )
    assert processor._generate_clean_id(function) == "Model.relu"
    assert processor._generate_clean_id(function) == "Model.relu_1"
    root.parents = [root]
    assert processor._get_module_hierarchy_with_ids(root) == [("Block", id(root))]


def test_visual_graph_fallback_parses_nodes_edges_and_bad_shape():
    source = """
    1 [label=<
      <TD>input-tensor<BR/>depth:1</TD><TD>(2, 3)</TD>
    ]
    2 [label=<
      <TD>output-tensor<BR/>depth:1</TD><TD>(2, 3)</TD>
    ]
    1 -> 2
    """
    processor = TorchviewProcessor()
    nodes = processor._extract_from_visual_graph(SimpleNamespace(source=source))
    assert len(nodes) == 2
    assert nodes[0].output_nodes == [nodes[1].node_id]
    assert processor._extract_from_visual_graph(object()) == []
    assert (
        processor._parse_node_definition(
            "3", "<TD>input-tensor<BR/>depth:1</TD><TD>(bad)</TD>"
        )
        is None
    )


def test_model_parameter_matching_supports_direct_and_shape_based_modules():
    processor = TorchviewProcessor()
    processor._reset_state()
    direct_module = nn.ReLU()
    module_node = _node("ModuleNode", name="relu", module=direct_module)
    processor._original_to_clean_id["module"] = "Model.relu"
    direct_info = NodeInfo(
        node_id="Model.relu",
        type="relu",
        node_class="ModuleNode",
        input_shapes=[[2]],
        output_shapes=[[2]],
    )
    processor._apply_model_parameters(
        [direct_info], nn.Sequential(nn.ReLU()), {"module": module_node}
    )
    assert direct_info.module_args["module_type"] == "ReLU"

    linear_info = NodeInfo(
        node_id="Model.linear",
        type="linear",
        node_class="FunctionNode",
        input_shapes=[[2, 3]],
        output_shapes=[[2, 4]],
    )
    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(3, 4))
    processor._reset_state()
    processor._apply_model_parameters([linear_info], model)
    assert linear_info.module_args["in_features"] == 3
    assert linear_info.module_args["out_features"] == 4

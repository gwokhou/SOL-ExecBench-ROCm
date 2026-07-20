from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn.functional as functional

from solar.verification import einsum as verification
from solar.verification.einsum import EinsumExecutionError, EinsumGraphExecutor


def _metadata(value: Any) -> tuple[list[list[int]], list[str]]:
    values = list(value) if isinstance(value, (tuple, list)) else [value]
    return [list(item.shape) for item in values], [str(item.dtype) for item in values]


def _execute(
    target: str,
    tensors: tuple[torch.Tensor, ...],
    *,
    arguments: list[Any] | None = None,
    kwargs: dict[str, Any] | None = None,
    expected: Any,
    effects: dict[str, Any] | None = None,
    overload: str = "default",
    check_shapes: bool = True,
) -> Any:
    output_shapes, output_dtypes = _metadata(expected)
    starts = {}
    names = []
    for index, tensor in enumerate(tensors):
        name = f"input_{index}"
        names.append(name)
        starts[f"start_{index}"] = {
            "type": "start",
            "semantic_op": {
                "kind": "input",
                "target": "input",
                "arguments": [],
                "kwargs": {},
            },
            "tensor_names": {"inputs": [], "outputs": [name]},
            "tensor_shapes": {"inputs": [], "outputs": [list(tensor.shape)]},
            "tensor_dtypes": {"inputs": [], "outputs": [str(tensor.dtype)]},
        }
    output_names = [f"output_{index}" for index in range(len(output_shapes))]
    operation = {
        "type": target,
        "semantic_op": {
            "kind": "aten",
            "target": target,
            "overload": overload,
            "arguments": (
                arguments
                if arguments is not None
                else [{"tensor": index} for index in range(len(tensors))]
            ),
            "kwargs": kwargs or {},
            "effects": effects
            or {
                "mutates": [],
                "aliases": [],
                "atomic": False,
                "opaque_library_call": False,
            },
        },
        "tensor_names": {"inputs": names, "outputs": output_names},
        "tensor_shapes": {
            "inputs": [list(tensor.shape) for tensor in tensors],
            "outputs": output_shapes,
        },
        "tensor_dtypes": {
            "inputs": [str(tensor.dtype) for tensor in tensors],
            "outputs": output_dtypes,
        },
    }
    graph = {
        "schema_version": 3,
        "layers": {**starts, "operation": operation},
        "outputs": output_names,
    }
    return EinsumGraphExecutor(graph, check_shapes=check_shapes)(
        *[tensor.clone() for tensor in tensors]
    )


@pytest.mark.parametrize(
    ("target", "function"),
    [
        ("add", torch.add),
        ("sub", torch.sub),
        ("mul", torch.mul),
        ("div", torch.div),
        ("pow", torch.pow),
        ("maximum", torch.maximum),
        ("minimum", torch.minimum),
    ],
)
def test_binary_dispatch(target, function) -> None:
    left = torch.tensor([1.0, 2.0])
    right = torch.tensor([2.0, 2.0])
    expected = function(left, right)
    torch.testing.assert_close(
        _execute(target, (left, right), expected=expected), expected
    )


@pytest.mark.parametrize(
    ("target", "function"),
    [
        ("abs", torch.abs),
        ("cos", torch.cos),
        ("exp", torch.exp),
        ("gelu", functional.gelu),
        ("hardsigmoid", functional.hardsigmoid),
        ("hardswish", functional.hardswish),
        ("log", torch.log),
        ("mish", functional.mish),
        ("neg", torch.neg),
        ("relu", functional.relu),
        ("rsqrt", torch.rsqrt),
        ("sigmoid", torch.sigmoid),
        ("silu", functional.silu),
        ("sin", torch.sin),
        ("sqrt", torch.sqrt),
        ("square", torch.square),
        ("tanh", torch.tanh),
    ],
)
def test_unary_dispatch(target, function) -> None:
    value = torch.tensor([1.0, 2.0])
    expected = function(value)
    torch.testing.assert_close(_execute(target, (value,), expected=expected), expected)


def test_einsum_and_matrix_dispatch() -> None:
    left = torch.arange(6.0).reshape(2, 3)
    right = torch.arange(12.0).reshape(3, 4)
    expected = left @ right
    layer = {
        "type": "matmul",
        "semantic_op": {
            "kind": "einsum",
            "target": "einsum",
            "equation": "A0B,BC->A0C",
            "arguments": [{"tensor": 0}, {"tensor": 1}],
            "kwargs": {},
            "effects": {"mutates": [], "aliases": []},
        },
        "tensor_names": {"inputs": ["left", "right"], "outputs": ["output"]},
        "tensor_shapes": {
            "inputs": [[2, 3], [3, 4]],
            "outputs": [[2, 4]],
        },
        "tensor_dtypes": {
            "inputs": ["torch.float32", "torch.float32"],
            "outputs": ["torch.float32"],
        },
    }
    starts = {}
    for name, tensor in (("left", left), ("right", right)):
        starts[name] = {
            "type": "start",
            "semantic_op": {"kind": "input"},
            "tensor_names": {"inputs": [], "outputs": [name]},
            "tensor_shapes": {"inputs": [], "outputs": [list(tensor.shape)]},
            "tensor_dtypes": {"inputs": [], "outputs": [str(tensor.dtype)]},
        }
    graph = {"schema_version": 3, "layers": {**starts, "op": layer}}
    torch.testing.assert_close(EinsumGraphExecutor(graph)(left, right), expected)
    torch.testing.assert_close(
        _execute("matmul", (left, right), expected=expected), expected
    )

    bias = torch.ones(2, 4)
    addmm = torch.addmm(bias, left, right)
    actual = _execute("addmm", (bias, left, right), expected=addmm)
    torch.testing.assert_close(actual, addmm)


def test_mutation_mask_identity_and_dtype_dispatch() -> None:
    left = torch.ones(2)
    right = torch.full((2,), 2.0)
    mutated = _execute(
        "add",
        (left, right),
        expected=left + right,
        effects={"mutates": [0], "aliases": [], "atomic": False},
    )
    torch.testing.assert_close(mutated, left + right)

    mask = torch.tensor([True, False])
    expected = left.masked_fill(mask, -2.0)
    actual = _execute(
        "masked_fill",
        (left, mask),
        arguments=[{"tensor": 0}, {"tensor": 1}, {"value": -2.0}],
        expected=expected,
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(_execute("identity", (left,), expected=left), left)

    half = left.half()
    actual = _execute(
        "to",
        (left,),
        arguments=[{"tensor": 0}, {"dtype": "float16"}],
        expected=half,
    )
    assert actual.dtype == torch.float16
    assert _execute("long", (left,), expected=left.long()).dtype == torch.int64
    assert _execute("type_as", (left, half), expected=left.type_as(half)).dtype == (
        torch.float16
    )
    assert (
        _execute("clone", (left,), expected=left.clone()).data_ptr() != left.data_ptr()
    )
    assert torch.equal(_execute("detach", (left,), expected=left.detach()), left)


def test_reduction_softmax_and_cumulative_dispatch() -> None:
    value = torch.arange(6.0).reshape(2, 3)
    for target, expected in (
        ("sum", value.sum(dim=1)),
        ("mean", value.mean(dim=1)),
        ("prod", value.prod(dim=1)),
        ("amax", value.amax(dim=1)),
        ("argmax", value.argmax(dim=1)),
        ("logsumexp", value.logsumexp(dim=1)),
    ):
        arguments = [{"tensor": 0}, {"value": 1}]
        actual = _execute(target, (value,), arguments=arguments, expected=expected)
        torch.testing.assert_close(actual, expected)
    for target, expected in (
        ("softmax", torch.softmax(value, dim=1)),
        ("log_softmax", torch.log_softmax(value, dim=1)),
    ):
        actual = _execute(
            target,
            (value,),
            arguments=[{"tensor": 0}, {"value": 1}],
            expected=expected,
        )
        torch.testing.assert_close(actual, expected)
    cumsum = torch.cumsum(value, dim=1)
    torch.testing.assert_close(
        _execute(
            "cumsum",
            (value,),
            arguments=[{"tensor": 0}, {"value": 1}],
            expected=cumsum,
        ),
        cumsum,
    )


def test_view_and_layout_dispatch() -> None:
    value = torch.arange(6.0).reshape(2, 3)
    reshaped = value.reshape(3, 2)
    torch.testing.assert_close(
        _execute(
            "reshape",
            (value,),
            arguments=[{"tensor": 0}],
            kwargs={"shape": [3, 2]},
            expected=reshaped,
        ),
        reshaped,
    )
    viewed = value.view(3, 2)
    torch.testing.assert_close(
        _execute(
            "view",
            (value,),
            arguments=[{"tensor": 0}, {"value": [3, 2]}],
            expected=viewed,
        ),
        viewed,
    )
    for target, arguments, expected in (
        ("flatten", [{"tensor": 0}], torch.flatten(value)),
        ("contiguous", [{"tensor": 0}], value.contiguous()),
        ("unsqueeze", [{"tensor": 0}, {"value": 0}], value.unsqueeze(0)),
        ("permute", [{"tensor": 0}, {"value": [1, 0]}], value.permute(1, 0)),
        ("repeat", [{"tensor": 0}, {"value": 2}, {"value": 1}], value.repeat(2, 1)),
        ("expand", [{"tensor": 0}, {"value": 2}, {"value": 3}], value.expand(2, 3)),
    ):
        torch.testing.assert_close(
            _execute(target, (value,), arguments=arguments, expected=expected), expected
        )
    transposed = value.transpose(0, 1)
    torch.testing.assert_close(
        _execute(
            "transpose",
            (value,),
            arguments=[{"tensor": 0}, {"value": 0}, {"value": 1}],
            expected=transposed,
        ),
        transposed,
    )
    square = torch.arange(4.0).reshape(2, 2)
    torch.testing.assert_close(
        _execute("transpose", (square,), expected=square.t()), square.t()
    )


def test_concat_split_and_indexing_dispatch() -> None:
    left = torch.ones(2, 1)
    right = torch.zeros(2, 1)
    cat = torch.cat((left, right), dim=1)
    actual = _execute(
        "cat",
        (left, right),
        arguments=[[{"tensor": 0}, {"tensor": 1}], {"value": 1}],
        expected=cat,
    )
    torch.testing.assert_close(actual, cat)
    stacked = torch.stack((left, right), dim=0)
    torch.testing.assert_close(
        _execute(
            "stack",
            (left, right),
            arguments=[{"tensor": 0}, {"tensor": 1}],
            kwargs={"dim": 0},
            expected=stacked,
        ),
        stacked,
    )
    split = torch.split(cat, 1, dim=1)
    result = _execute(
        "split",
        (cat,),
        arguments=[{"tensor": 0}, {"value": 1}],
        kwargs={"dim": 1},
        expected=split,
    )
    assert len(result) == 2

    index = torch.tensor([[0, 1]])
    gathered = torch.gather(cat, 0, index)
    torch.testing.assert_close(
        _execute(
            "gather",
            (cat, index),
            arguments=[{"tensor": 0}, {"value": 0}, {"tensor": 1}],
            expected=gathered,
        ),
        gathered,
    )
    selected = torch.index_select(cat, 1, torch.tensor([1]))
    torch.testing.assert_close(
        _execute(
            "index_select",
            (cat, torch.tensor([1])),
            arguments=[{"tensor": 0}, {"value": 1}, {"tensor": 1}],
            expected=selected,
        ),
        selected,
    )
    sliced = cat[:, :1]
    index_argument = [
        {"slice": [None, None, None]},
        {"slice": [None, 1, None]},
    ]
    torch.testing.assert_close(
        _execute(
            "getitem",
            (cat,),
            arguments=[{"tensor": 0}, index_argument],
            expected=sliced,
        ),
        sliced,
    )
    torch.testing.assert_close(
        _execute(
            "slice",
            (cat,),
            arguments=[{"tensor": 0}],
            kwargs={"dim": 1, "start": 0, "end": 1, "step": 1},
            expected=sliced,
        ),
        sliced,
    )


def test_library_quantization_and_aten_fallback_dispatch() -> None:
    value = torch.arange(6.0).reshape(2, 3)
    weight = torch.ones(4, 3)
    bias = torch.ones(4)
    linear = functional.linear(value, weight, bias)
    torch.testing.assert_close(
        _execute("linear", (value, weight, bias), expected=linear), linear
    )

    image = torch.ones(1, 1, 3, 3)
    kernel = torch.ones(1, 1, 1, 1)
    conv = functional.conv2d(image, kernel)
    torch.testing.assert_close(_execute("conv2d", (image, kernel), expected=conv), conv)

    normalized = functional.layer_norm(value, (3,))
    torch.testing.assert_close(
        _execute(
            "layer_norm",
            (value,),
            arguments=[{"tensor": 0}, {"value": [3]}],
            expected=normalized,
        ),
        normalized,
    )
    indices = torch.tensor([0, 1])
    table = torch.arange(12.0).reshape(3, 4)
    embedded = functional.embedding(indices, table)
    torch.testing.assert_close(
        _execute("embedding", (indices, table), expected=embedded), embedded
    )
    clamped = torch.clamp(value, min=1.0, max=4.0)
    torch.testing.assert_close(
        _execute(
            "clamp",
            (value,),
            kwargs={"min": 1.0, "max": 4.0},
            expected=clamped,
        ),
        clamped,
    )
    reciprocal = torch.ops.aten.reciprocal.default(value + 1)
    torch.testing.assert_close(
        _execute("reciprocal", (value + 1,), expected=reciprocal), reciprocal
    )


def test_executor_rejects_invalid_graph_and_runtime_contracts() -> None:
    with pytest.raises(EinsumExecutionError, match="schema_version"):
        EinsumGraphExecutor({"schema_version": 0, "layers": {}})
    with pytest.raises(EinsumExecutionError, match="unsupported extended einsum"):
        verification._torch_equation("A(P+R)->A")
    with pytest.raises(EinsumExecutionError, match="explicit transpose dimensions"):
        value = torch.ones(2, 2, 2)
        _execute("transpose", (value,), expected=value)
    with pytest.raises(EinsumExecutionError, match="unsupported exact operation"):
        value = torch.ones(1)
        _execute("definitely_missing", (value,), expected=value)


def test_argument_decoder_rejects_bad_tensor_dtype_and_mapping() -> None:
    value = torch.ones(1)
    with pytest.raises(EinsumExecutionError, match="outside its input metadata"):
        _execute(
            "identity",
            (value,),
            arguments=[{"tensor": 3}],
            expected=value,
        )
    with pytest.raises(EinsumExecutionError, match="invalid dtype"):
        _execute(
            "to",
            (value,),
            arguments=[{"tensor": 0}, {"dtype": "not_a_dtype"}],
            expected=value,
        )
    with pytest.raises(EinsumExecutionError, match="invalid semantic argument"):
        _execute(
            "identity",
            (value,),
            arguments=[{"tensor": 0}, {"unknown": 1}],
            expected=value,
        )

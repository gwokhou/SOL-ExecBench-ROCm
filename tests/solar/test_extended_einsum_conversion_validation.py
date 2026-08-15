from pathlib import Path

import pytest

from solar.graph.contracts import OperatorGraphArtifact, TensorSignature
from solar.ir.extended_einsum import conversion


def _operator(
    *,
    inputs: tuple[tuple[int, TensorSignature], ...] = (),
    outputs: tuple[TensorSignature, ...] = (),
) -> OperatorGraphArtifact:
    return OperatorGraphArtifact(
        path=Path("operator_graph.yaml"),
        source_inputs=inputs,
        used_source_indices=tuple(index for index, _ in inputs),
        reference_outputs=outputs,
    )


def _start(
    shape: list[int] | None = None,
    source_input_index: int | None = None,
) -> dict:
    layer = {
        "type": "start",
        "tensor_shapes": {"outputs": [] if shape is None else [shape]},
        "tensor_dtypes": {"outputs": ["torch.float32"]},
    }
    if source_input_index is not None:
        layer["source_input_index"] = source_input_index
    return layer


def test_validate_rejects_a_different_ir_kind() -> None:
    with pytest.raises(ValueError, match="not extended-einsum IR"):
        conversion.validate_extended_einsum_graph({"ir_kind": "aten"})


def test_input_binding_rejects_arity_and_incomplete_metadata() -> None:
    signature = TensorSignature(shape=(2, 3), dtype="torch.float32")
    operator = _operator(inputs=((0, signature),))

    with pytest.raises(RuntimeError, match=r"observed=.*starts=0"):
        conversion._bind_inputs({"layers": {}}, operator)

    with pytest.raises(
        RuntimeError, match="does not match graph input metadata"
    ):
        conversion._bind_inputs(
            {"layers": {"input": _start(source_input_index=0)}},
            operator,
        )


def test_input_binding_rejects_missing_exact_source_indices() -> None:
    signature = TensorSignature(shape=(2, 3), dtype="torch.float32")
    operator = _operator(inputs=((0, signature), (1, signature)))
    graph = {
        "layers": {
            "left": _start([2, 3]),
            "right": _start([2, 3]),
        },
    }

    with pytest.raises(RuntimeError, match="lacks an exact source index"):
        conversion._bind_inputs(graph, operator)


def test_input_binding_uses_source_indices_for_identical_signatures() -> None:
    signature = TensorSignature(shape=(2, 3), dtype="torch.float32")
    operator = _operator(inputs=((0, signature), (1, signature)))
    graph = {
        "layers": {
            "left": _start([2, 3], 0),
            "right": _start([2, 3], 1),
        },
    }

    assert conversion._bind_inputs(graph, operator) == [0, 1]


def test_output_binding_rejects_arity_and_signature_mismatches() -> None:
    expected = (TensorSignature(shape=(2, 3), dtype="torch.float32"),)

    with pytest.raises(RuntimeError, match="output arity"):
        conversion._bind_outputs({"layers": {}}, {"layers": {}}, expected)

    graph = {
        "layers": {
            "producer": {
                "tensor_names": {"outputs": ["result"]},
                "tensor_shapes": {"outputs": [[4, 5]]},
                "tensor_dtypes": {"outputs": ["torch.float32"]},
            },
        },
    }
    traced = {
        "layers": {
            "output": {
                "type": "output-tensor",
                "connections": {"inputs": ["producer"]},
            },
        },
    }
    with pytest.raises(RuntimeError, match="does not match reference"):
        conversion._bind_outputs(graph, traced, expected)


@pytest.mark.parametrize(
    ("producer", "message"),
    [
        (None, "cannot bind exact"),
        (
            {
                "tensor_names": {"outputs": ["first", "second"]},
                "tensor_shapes": {"outputs": [[2, 3]]},
                "tensor_dtypes": {"outputs": ["torch.float32"]},
            },
            "lacks an exact producer output slot",
        ),
    ],
)
def test_output_candidates_require_one_complete_producer(
    producer: dict | None,
    message: str,
) -> None:
    graph = {"layers": {}} if producer is None else {"layers": {"p": producer}}
    traced = {
        "layers": {
            "output": {
                "type": "output-tensor",
                "connections": {"inputs": ["p"]},
            },
        },
    }

    with pytest.raises(RuntimeError, match=message):
        conversion._output_candidates(graph, traced)


def test_output_candidates_preserve_independent_multi_output_slots() -> None:
    graph = {
        "layers": {
            "topk": {
                "tensor_names": {"outputs": ["values", "indices"]},
                "tensor_shapes": {"outputs": [[2, 3], [2, 3]]},
                "tensor_dtypes": {
                    "outputs": ["torch.float32", "torch.int64"],
                },
            },
        },
    }
    traced = {
        "layers": {
            "values": {
                "type": "output-tensor",
                "connections": {"inputs": ["topk"]},
                "module_args": {"producer_output_slot": 0},
            },
            "indices": {
                "type": "output-tensor",
                "connections": {"inputs": ["topk"]},
                "module_args": {"producer_output_slot": 1},
            },
        },
    }

    assert conversion._output_candidates(graph, traced) == [
        ("values", [2, 3], "torch.float32"),
        ("indices", [2, 3], "torch.int64"),
    ]

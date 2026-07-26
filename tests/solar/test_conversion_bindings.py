from pathlib import Path

import pytest
import yaml

from solar.einsum.bindings import bind_inputs
from solar.einsum.conversion import convert_operator_graph
from solar.graph.extraction import OperatorGraphArtifact, TensorSignature


def _start(name: str, shape: list[int], source_input_index: int) -> dict:
    return {
        "type": "start",
        "source_input_index": source_input_index,
        "source_binding": "torchview_input_order",
        "tensor_names": {"inputs": [], "outputs": [name]},
        "tensor_shapes": {"inputs": [], "outputs": [shape]},
        "tensor_dtypes": {"inputs": [], "outputs": ["torch.float32"]},
    }


def test_source_binding_orders_traced_and_recovered_inputs_independently() -> None:
    graph = {
        "layers": {
            "start": _start("start.Output", [2, 3], 0),
            "start_1": _start("start_1.Output", [4, 3], 3),
            "start_2": _start("start_2.Output", [3], 4),
            "normalize": {
                "type": "layer_norm",
                "tensor_names": {
                    "inputs": [
                        "start.Output",
                        "normalize.Weight",
                        "normalize.Weight_1",
                    ],
                    "outputs": ["normalize.Output"],
                },
                "tensor_shapes": {
                    "inputs": [[2, 3], [3], [3]],
                    "outputs": [[2, 3]],
                },
                "tensor_dtypes": {
                    "inputs": ["torch.float32"] * 3,
                    "outputs": ["torch.float32"],
                },
            },
            "consumer": {
                "type": "add",
                "tensor_names": {
                    "inputs": [
                        "normalize.Output",
                        "start_1.Output",
                        "start_2.Output",
                    ],
                    "outputs": ["consumer.Output"],
                },
                "tensor_shapes": {
                    "inputs": [[2, 3], [4, 3], [3]],
                    "outputs": [[2, 3]],
                },
                "tensor_dtypes": {
                    "inputs": ["torch.float32"] * 3,
                    "outputs": ["torch.float32"],
                },
            },
        }
    }
    operator = OperatorGraphArtifact(
        path=Path("operator_graph.yaml"),
        source_inputs=(
            (0, TensorSignature((2, 3), "torch.float32")),
            (1, TensorSignature((3,), "torch.float32")),
            (2, TensorSignature((3,), "torch.float32")),
            (3, TensorSignature((4, 3), "torch.float32")),
            (4, TensorSignature((3,), "torch.float32")),
        ),
        used_source_indices=(0, 1, 2, 3, 4),
        reference_outputs=(TensorSignature((2, 3), "torch.float32"),),
    )

    assert bind_inputs(graph, operator) == [0, 3, 4, 1, 2]


def test_schema_v3_graph_requires_make_fx_provenance(tmp_path: Path) -> None:
    operator_path = tmp_path / "operator_graph.yaml"
    operator_path.write_text(
        yaml.safe_dump({"schema_version": 3, "layers": {}, "outputs": []}),
        encoding="utf-8",
    )
    operator = OperatorGraphArtifact(
        path=operator_path,
        source_inputs=(),
        used_source_indices=(),
        reference_outputs=(),
    )

    with pytest.raises(RuntimeError, match="provenance is not trusted"):
        convert_operator_graph(operator, output_dir=tmp_path)

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
        "tensor_names": {"inputs": [], "outputs": [name]},
        "tensor_shapes": {"inputs": [], "outputs": [shape]},
        "tensor_dtypes": {"inputs": [], "outputs": ["torch.float32"]},
    }


def _operator(path: Path, *indices: int) -> OperatorGraphArtifact:
    return OperatorGraphArtifact(
        path=path,
        source_inputs=tuple(
            (index, TensorSignature((2, 3), "torch.float32")) for index in indices
        ),
        used_source_indices=indices,
        reference_outputs=(),
    )


def test_bind_inputs_uses_recorded_source_argument_indices() -> None:
    graph = {
        "layers": {
            "second": _start("second", [2, 3], 3),
            "first": _start("first", [2, 3], 0),
        }
    }

    assert bind_inputs(graph, _operator(Path("operator_graph.yaml"), 0, 3)) == [3, 0]


def test_bind_inputs_rejects_missing_or_duplicate_provenance() -> None:
    graph = {
        "layers": {
            "first": _start("first", [2, 3], 0),
            "duplicate": _start("duplicate", [2, 3], 0),
        }
    }

    with pytest.raises(RuntimeError, match="provenance is invalid"):
        bind_inputs(graph, _operator(Path("operator_graph.yaml"), 0, 3))


def test_conversion_requires_canonical_schema_and_provenance(tmp_path: Path) -> None:
    operator_path = tmp_path / "operator_graph.yaml"
    operator_path.write_text(
        yaml.safe_dump({"schema_version": 3, "layers": {}, "outputs": []}),
        encoding="utf-8",
    )
    operator = _operator(operator_path)

    with pytest.raises(RuntimeError, match="provenance is not trusted"):
        convert_operator_graph(operator, output_dir=tmp_path)

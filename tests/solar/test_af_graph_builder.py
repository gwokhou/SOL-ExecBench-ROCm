from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from solar.einsum import af_graph_builder as afb


def _start(name: str = "start", shape: list[int] | None = None) -> dict:
    shape = shape or [2, 3]
    labels = [chr(ord("A") + index) for index in range(len(shape))]
    return {
        "type": "start",
        "is_real_einsum": False,
        "operands": {"Output": labels},
        "tensor_names": {"inputs": [], "outputs": [f"{name}.Output"]},
        "tensor_shapes": {"inputs": [], "outputs": [shape]},
        "tensor_dtypes": {"inputs": [], "outputs": ["torch.float16"]},
        "tensor_types": {"inputs": [], "outputs": ["input"]},
        "connections": {"inputs": [], "outputs": []},
    }


def _matmul() -> dict:
    return {
        "type": "matmul",
        "is_real_einsum": True,
        "operands": {
            "Input": ["M", "K"],
            "Weight": ["K", "N"],
            "Output": ["M", "N"],
        },
        "tensor_names": {
            "inputs": ["start.Output", "matmul.Weight"],
            "outputs": ["matmul.Output"],
        },
        "tensor_shapes": {
            "inputs": [[2, 3], [3, 4]],
            "outputs": [[2, 4]],
        },
        "tensor_dtypes": {
            "inputs": ["torch.float16", "torch.float16"],
            "outputs": ["torch.float32"],
        },
        "tensor_types": {"inputs": ["input", "weight"], "outputs": ["output"]},
        "connections": {"inputs": ["start"], "outputs": []},
    }


def test_public_builder_emits_consistent_workload_and_yaml(tmp_path: Path):
    graph = {
        "model_name": "matrix-model",
        "layers": {"matmul": _matmul(), "start": _start()},
    }
    af = afb.build_af_graph_from_dict(graph)
    assert [item["name"] for item in af["workload"]["einsums"]] == [
        "start",
        "matmul",
    ]
    assert af["workload"]["bits_per_value"]["All"] == 32
    assert af["workload"]["rank_sizes"]

    source = tmp_path / "einsum.yaml"
    target = tmp_path / "nested" / "af.yaml"
    source.write_text(yaml.safe_dump(graph), encoding="utf-8")
    loaded = afb.build_af_graph_from_yaml(source, target)
    assert loaded["workload"]["rank_sizes"] == af["workload"]["rank_sizes"]
    written = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert written == {
        key: value for key, value in loaded.items() if not key.startswith("_")
    }
    assert afb.build_af_einsum_graph is afb.build_af_graph_from_yaml
    with pytest.raises(ValueError, match="no layers"):
        afb.build_af_graph_from_dict({})


def test_identifier_role_atom_and_dtype_helpers():
    assert afb._is_output_role("Output_2")
    assert not afb._is_output_role("Input")
    assert afb._is_input_role("Weight_1")
    assert not afb._is_input_role("Target")
    assert afb._parse_atoms("P+R") == ["P", "R"]
    assert afb._parse_atoms("P") == ["P"]
    assert afb._sanitize("3.Model-weight") == "_3_Model_weight"
    assert afb._sanitize(cast(Any, 3)) == 3
    assert afb._bits_from_dtype("torch.bfloat16") == 16
    assert afb._bits_from_dtype("complex128") == 128
    assert afb._bits_from_dtype("unknown") is None
    assert afb._bits_from_dtype(cast(Any, None)) is None


def test_union_find_is_stable_and_compresses_paths():
    uf = afb.UnionFind()
    a = afb.AxisKey("layer", "Input", 0)
    b = afb.AxisKey("layer", "Weight", 0)
    c = afb.AxisKey("layer", "Output", 0)
    uf.add(a)
    uf.add(a)
    assert uf.union(a, b) == a
    assert uf.union(c, b) == a
    assert uf.union(a, c) == a
    assert uf.find(c) == a


@pytest.mark.parametrize(
    ("op_type", "in_dims", "in_shape", "out_dims", "out_shape", "expected"),
    [
        ("contiguous", ["A", "B"], [2, 3], ["A", "B"], [2, 3], ([0, 1], [[0], [1]])),
        ("permute", ["A", "B"], [2, 3], ["B", "A"], [3, 2], ([1, 0], [[1], [0]])),
        ("transpose", ["A", "B"], [2, 3], ["A", "B"], [3, 2], ([1, 0], [[1], [0]])),
        (
            "squeeze",
            ["A", "B", "C"],
            [2, 1, 3],
            ["A", "C"],
            [2, 3],
            ([0, 2], [[0], [], [1]]),
        ),
        ("squeeze", ["A", "B"], [2, 4], ["A"], [2], None),
        (
            "unsqueeze",
            ["A", "B"],
            [2, 3],
            ["A", "U", "B"],
            [2, 1, 3],
            ([0, None, 1], [[0], [2]]),
        ),
        ("unsqueeze", ["A"], [2], ["A", "B"], [2, 4], None),
        ("expand", ["A", "B"], [2, 1], ["A", "B"], [2, 8], ([0, 1], [[0], [1]])),
        ("__getitem__", ["A", "B"], [2, 3], ["A"], [2], None),
        (
            "view",
            ["A", "B"],
            [2, 3],
            ["A", "U", "B"],
            [2, 1, 3],
            ([0, None, 1], [[0], [2]]),
        ),
        ("reshape", ["A", "B"], [2, 3], ["C"], [6], None),
        ("unrelated", ["A"], [2], ["B", "C"], [1, 2], None),
    ],
)
def test_shape_position_mapping_matrix(
    op_type, in_dims, in_shape, out_dims, out_shape, expected
):
    assert (
        afb._derive_pos_mapping(
            "shape", {"type": op_type}, in_dims, in_shape, out_dims, out_shape
        )
        == expected
    )


def _shape_layer(
    name: str,
    predecessor: str,
    input_tensor: str,
    output_tensor: str,
    *,
    op_type: str,
    input_shape: list[int],
    output_shape: list[int],
) -> dict:
    in_dims = [f"I{i}" for i in range(len(input_shape))]
    out_dims = [f"O{i}" for i in range(len(output_shape))]
    return {
        "type": op_type,
        "is_real_einsum": False,
        "operands": {"Input": in_dims, "Output": out_dims},
        "tensor_names": {"inputs": [input_tensor], "outputs": [output_tensor]},
        "tensor_shapes": {"inputs": [input_shape], "outputs": [output_shape]},
        "tensor_dtypes": {"inputs": ["torch.float16"], "outputs": ["torch.float16"]},
        "tensor_types": {"inputs": ["input"], "outputs": ["output"]},
        "connections": {"inputs": [predecessor], "outputs": []},
    }


def test_shape_op_elision_rewrites_a_chained_alias():
    start = _start(shape=[2, 3])
    start["connections"]["outputs"] = ["transpose"]
    transpose = _shape_layer(
        "transpose",
        "start",
        "start.Output",
        "transpose.Output",
        op_type="transpose",
        input_shape=[2, 3],
        output_shape=[3, 2],
    )
    transpose["connections"]["outputs"] = ["unsqueeze"]
    unsqueeze = _shape_layer(
        "unsqueeze",
        "transpose",
        "transpose.Output",
        "unsqueeze.Output",
        op_type="unsqueeze",
        input_shape=[3, 2],
        output_shape=[1, 3, 2],
    )
    unsqueeze["connections"]["outputs"] = ["consumer"]
    consumer = {
        "type": "relu",
        "is_real_einsum": False,
        "operands": {"Input": ["X", "Y", "Z"], "Output": ["X", "Y", "Z"]},
        "tensor_names": {
            "inputs": ["unsqueeze.Output"],
            "outputs": ["consumer.Output"],
        },
        "tensor_shapes": {"inputs": [[1, 3, 2]], "outputs": [[1, 3, 2]]},
        "tensor_dtypes": {"inputs": ["torch.float16"], "outputs": ["torch.float16"]},
        "tensor_types": {"inputs": ["input"], "outputs": ["output"]},
        "connections": {"inputs": ["unsqueeze"], "outputs": []},
    }
    rewritten, diagnostics = afb._apply_shape_op_elision(
        {
            "start": start,
            "transpose": transpose,
            "unsqueeze": unsqueeze,
            "consumer": consumer,
        }
    )
    assert set(rewritten) == {"start", "consumer"}
    assert rewritten["consumer"]["connections"]["inputs"] == ["start"]
    assert rewritten["consumer"]["tensor_names"]["inputs"] == ["start.Output"]
    assert rewritten["consumer"]["tensor_shapes"]["inputs"] == [[2, 3]]
    assert diagnostics == []


def test_operand_normalization_synthesizes_multi_input_and_output_roles():
    layers = {
        "cat": {
            "operands": {"Input": ["A", "B"], "Output": ["A", "C"]},
            "tensor_shapes": {
                "inputs": [[2, 3], [2, 4], [2, 3]],
                "outputs": [[2, 10], [2]],
            },
            "tensor_types": {
                "inputs": ["input", "input", "weight"],
                "outputs": ["output", "output"],
            },
        }
    }
    diagnostics = afb._normalize_operands(layers)
    assert layers["cat"]["operands"]["Input_1"] == ["A", "B_s1"]
    assert "Weight_2" not in layers["cat"]["operands"]
    assert layers["cat"]["operands"]["Output_1"] == ["Output_1_d0"]
    assert len(diagnostics) == 2

    empty_shape = {
        "op": {
            "operands": {"Input": ["A"]},
            "tensor_shapes": {"inputs": [[2], []], "outputs": []},
            "tensor_types": {"inputs": ["input", "input"], "outputs": []},
        }
    }
    assert "no shape info" in afb._normalize_operands(empty_shape)[0]


def test_ghost_scalar_output_uses_smallest_rank_but_not_when_consumed():
    af: dict[str, Any] = {
        "workload": {
            "rank_sizes": {"R0": 8, "R1": 2},
            "einsums": [
                {
                    "name": "reduce",
                    "tensor_accesses": [
                        {"name": "input", "projection": {"R0": "r0", "R1": "r1"}},
                        {"name": "scalar", "projection": [], "output": True},
                    ],
                }
            ],
        }
    }
    afb._ghost_scalar_outputs(af)
    assert af["workload"]["einsums"][0]["tensor_accesses"][1]["projection"] == ["r1"]

    consumed: dict[str, Any] = deepcopy(af)
    consumed["workload"]["einsums"].append(
        {
            "name": "consumer",
            "tensor_accesses": [
                {"name": "scalar", "projection": []},
                {"name": "out", "projection": [], "output": True},
            ],
        }
    )
    consumed["workload"]["einsums"][0]["tensor_accesses"][1]["projection"] = []
    afb._ghost_scalar_outputs(consumed)
    assert consumed["workload"]["einsums"][0]["tensor_accesses"][1]["projection"] == []


def test_invariant_validation_reports_rank_and_provenance_errors():
    af = {
        "workload": {
            "rank_sizes": {"R0": 2},
            "einsums": [
                {
                    "name": "producer",
                    "tensor_accesses": [
                        {"name": "input", "projection": ["r0"]},
                        {"name": "shared", "projection": ["r0"], "output": True},
                    ],
                },
                {
                    "name": "consumer",
                    "tensor_accesses": [
                        {"name": "shared", "projection": {"R1": "r1"}},
                        {"name": "out", "projection": ["r0"], "output": True},
                    ],
                },
            ],
        }
    }
    with pytest.raises(RuntimeError, match="inconsistent rank tuples"):
        afb._validate_graph_invariants(af)


def test_topological_sort_is_stable_and_preserves_cycles():
    ordered = afb._topological_sort_layers(
        {
            "consumer": {"connections": {"inputs": ["producer"]}},
            "producer": {"connections": {"inputs": []}},
        }
    )
    assert list(ordered) == ["producer", "consumer"]
    cyclic = {
        "a": {"connections": {"inputs": ["b"]}},
        "b": {"connections": {"inputs": ["a"]}},
    }
    assert list(afb._topological_sort_layers(cyclic)) == ["a", "b"]

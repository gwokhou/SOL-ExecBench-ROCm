from __future__ import annotations

import csv
from copy import deepcopy
from pathlib import Path

import pytest

from solar.analysis import orojenesis


def _start(output: str) -> dict:
    return {
        "type": "start",
        "tensor_names": {"inputs": [], "outputs": [output]},
    }


def _matmul(
    layer_id: str,
    activation: str,
    weight: str,
    output: str,
    *,
    m: int = 2,
    k: int = 3,
    n: int = 4,
    batch: int | None = None,
) -> dict:
    del layer_id
    if batch is None:
        equation = "MK,KN->MN"
        activation_shape = [m, k]
        output_shape = [m, n]
    else:
        equation = "BMK,KN->BMN"
        activation_shape = [batch, m, k]
        output_shape = [batch, m, n]
    return {
        "type": "operation",
        "semantic_op": {
            "kind": "einsum",
            "equation": equation,
            "effects": {
                "mutates": False,
                "aliases": [],
                "atomic": False,
                "opaque_library_call": False,
            },
        },
        "tensor_names": {
            "inputs": [activation, weight],
            "outputs": [output],
        },
        "tensor_shapes": {
            "inputs": [activation_shape, [k, n]],
            "outputs": [output_shape],
        },
        "tensor_dtypes": {
            "inputs": ["float16", "float16"],
            "outputs": ["float16"],
        },
    }


def _view(input_name: str, output_name: str) -> dict:
    return {
        "type": "operation",
        "semantic_op": {
            "kind": "aten",
            "target": "transpose",
            "arguments": [{"value": 0}, {"value": 1}],
            "effects": {
                "mutates": False,
                "atomic": False,
                "opaque_library_call": False,
                "aliases": [{"input": 0, "output": 0, "conditional": False}],
            },
        },
        "tensor_names": {"inputs": [input_name], "outputs": [output_name]},
        "tensor_shapes": {"inputs": [[2, 4]], "outputs": [[4, 2]]},
        "tensor_dtypes": {"inputs": ["float16"], "outputs": ["float16"]},
    }


def _linear_layers(*, batched: bool = False) -> dict[str, dict]:
    batch = 2 if batched else None
    first_m = 3 if batched else 2
    return {
        "x": _start("x"),
        "w0": _start("w0"),
        "w1": _start("w1"),
        "mm0": _matmul("mm0", "x", "w0", "hidden", m=first_m, k=3, n=4, batch=batch),
        "mm1": _matmul(
            "mm1",
            "hidden",
            "w1",
            "result",
            m=first_m,
            k=4,
            n=5,
            batch=batch,
        ),
    }


def _write_mapping(
    path: Path,
    *,
    buffer_bytes: int = 64,
    input_util: int = 16,
    output_util: int = 16,
    accesses: tuple[float, float, float] = (2, 3, 1),
    mapping: str = "mapping",
) -> Path:
    row: list[object] = [0] * 24
    row[0] = buffer_bytes
    row[2] = sum(accesses)
    row[3] = mapping
    row[5] = 120
    row[6] = 8
    row[10] = input_util
    row[11] = output_util
    row[21], row[22], row[23] = accesses
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        csv.writer(handle).writerow(row)
    return path


def test_problem_architecture_mapper_and_capacity_helpers():
    layer = _matmul("mm", "x", "w", "y")
    problem = orojenesis.OrojenesisRunner.problem_for_layer(layer)
    assert problem["problem"]["instance"] == {"A": 2, "B": 3, "C": 4}
    assert problem["problem"]["shape"]["dimensions"] == ["A", "B", "C"]
    assert (
        orojenesis.OrojenesisRunner.architecture(16)["architecture"]["subtree"][0][
            "local"
        ][0]["attributes"]["word-bits"]
        == 16
    )
    assert (
        orojenesis.OrojenesisRunner.multi_architecture(8)["architecture"]["version"]
        == 0.2
    )
    assert (
        orojenesis.OrojenesisRunner.mapper_config(["A", "B"], ["Input0"])[
            "mapspace_constraints"
        ][0]["permutation"]
        == "AB"
    )
    assert orojenesis.multi_einsum_row_tiles(12) == [1, 2, 3, 4, 6, 12]
    assert orojenesis.select_capacity_point(
        [
            {"buffer_bytes": 4, "dram_bytes": 20},
            {"buffer_bytes": 8, "dram_bytes": 10},
        ],
        6,
    ) == {"buffer_bytes": 4, "dram_bytes": 20}
    assert orojenesis.select_capacity_point([], 6) is None

    roles = [orojenesis.multi_einsum_mapper_role(index, 5) for index in range(5)]
    assert roles == ["first", "second", "middle", "middle", "last"]
    assert orojenesis.multi_einsum_mapper_role(1, 2) == "second_last"
    for role in {"first", "second", "middle", "last", "second_last"}:
        config = orojenesis.OrojenesisRunner.multi_mapper_config(2, role=role)
        assert config["mapspace_constraints"][-1]["factors"] == "M=2"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda layer: layer["semantic_op"].update(kind="aten"), "exact einsum"),
        (
            lambda layer: layer["semantic_op"].update(equation="MK,KN->M"),
            "output shape",
        ),
        (
            lambda layer: layer["tensor_shapes"]["inputs"][1].__setitem__(0, 9),
            "inconsistent dimension",
        ),
        (
            lambda layer: layer["tensor_shapes"].update(outputs=[]),
            "operand arity",
        ),
    ],
)
def test_problem_for_layer_rejects_invalid_metadata(mutation, message):
    layer = _matmul("mm", "x", "w", "y")
    mutation(layer)
    with pytest.raises(orojenesis.OrojenesisError, match=message):
        orojenesis.OrojenesisRunner.problem_for_layer(layer)


def test_mapper_and_divisor_validation():
    with pytest.raises(orojenesis.OrojenesisError, match="positive"):
        orojenesis.multi_einsum_row_tiles(0)
    with pytest.raises(orojenesis.OrojenesisError, match="chain position"):
        orojenesis.multi_einsum_mapper_role(0, 1)
    with pytest.raises(orojenesis.OrojenesisError, match="row tile"):
        orojenesis.OrojenesisRunner.multi_mapper_config(0, role="first")
    with pytest.raises(orojenesis.OrojenesisError, match="mapper role"):
        orojenesis.OrojenesisRunner.multi_mapper_config(1, role="unknown")


def test_curve_parsing_filters_rows_and_builds_pareto(tmp_path):
    path = tmp_path / "curve.csv"
    path.write_text(
        "bad\n64,1.0,20\n64,2.0,10\n128,3.0,12\n256,4.0,5\n",
        encoding="utf-8",
    )
    curve = orojenesis.OrojenesisRunner.parse_curve(path, word_bytes=2)
    assert [(item["buffer_bytes"], item["dram_bytes"]) for item in curve] == [
        (64, 20.0),
        (256, 10.0),
    ]
    with pytest.raises(orojenesis.OrojenesisError, match="missing OAVES"):
        orojenesis.OrojenesisRunner.parse_curve(tmp_path / "missing", word_bytes=2)
    path.write_text("bad,row,only\n", encoding="utf-8")
    with pytest.raises(orojenesis.OrojenesisError, match="no valid"):
        orojenesis.OrojenesisRunner.parse_curve(path, word_bytes=2)


def test_linear_chain_problem_discovery_and_roles():
    layers = _linear_layers()
    chain = [(name, layers[name]) for name in ("mm0", "mm1")]
    problem = orojenesis.multi_einsum_problem(chain)
    assert [item["id"] for item in problem["chain"]["layers"]] == ["mm0", "mm1"]
    assert orojenesis.find_multi_einsum_chains(layers) == [["mm0", "mm1"]]
    layer_problem = orojenesis.multi_einsum_layer_problem(problem["chain"]["layers"][0])
    assert layer_problem["problem"]["instance"] == {"M": 2, "K": 3, "N": 4}

    incomplete = deepcopy(layers)
    incomplete["sink"] = {
        "type": "operation",
        "tensor_names": {"inputs": ["result"], "outputs": ["sink"]},
    }
    assert orojenesis.find_multi_einsum_chains(incomplete) == []


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda chain: chain.pop(), "at least two"),
        (
            lambda chain: chain[1][1]["tensor_names"]["inputs"].__setitem__(0, "x"),
            "producer-consumer",
        ),
        (
            lambda chain: chain[1][1]["tensor_shapes"]["inputs"][0].__setitem__(0, 9),
            "shapes are inconsistent",
        ),
        (
            lambda chain: chain[1][1]["tensor_dtypes"]["outputs"].__setitem__(
                0, "float32"
            ),
            "one exact tensor dtype",
        ),
        (
            lambda chain: chain[0][1]["semantic_op"]["effects"].update(mutates=True),
            "observable effects",
        ),
    ],
)
def test_linear_chain_rejects_unsound_problems(mutation, message):
    layers = _linear_layers()
    chain = [(name, layers[name]) for name in ("mm0", "mm1")]
    mutation(chain)
    with pytest.raises(orojenesis.OrojenesisError, match=message):
        orojenesis.multi_einsum_problem(chain)


def test_mapping_record_parse_and_linear_composition(tmp_path):
    first = _write_mapping(tmp_path / "first.csv", mapping="first")
    second = _write_mapping(tmp_path / "second.csv", buffer_bytes=32, mapping="second")
    records = orojenesis.parse_multi_mapping_records(first, word_bytes=2)
    assert records[0]["dram_bytes"] == 12
    curve = orojenesis.compose_multi_einsum_curve(
        [[first], [second]], row_tiles=[2], word_bytes=2
    )
    assert curve[0]["buffer_bytes"] == 96
    assert curve[0]["mappings"] == ["first", "second"]

    serialized = tmp_path / "serialized.csv"
    with serialized.open("w", newline="") as handle:
        csv.writer(handle).writerow(
            [96, 5.0, curve[0]["dram_accesses_words"], '["first","second"]', 2]
        )
    assert (
        orojenesis.parse_multi_einsum_curve(serialized, word_bytes=2)[0]["row_tile"]
        == 2
    )


def test_mapping_and_composition_validation(tmp_path):
    missing = tmp_path / "missing.csv"
    with pytest.raises(orojenesis.OrojenesisError, match="missing multi-einsum"):
        orojenesis.parse_multi_mapping_records(missing, word_bytes=2)

    invalid = _write_mapping(tmp_path / "invalid.csv", accesses=(1, 1, 1))
    rows = list(csv.reader(invalid.open(newline="")))
    rows[0][2] = "99"
    with invalid.open("w", newline="") as handle:
        csv.writer(handle).writerows(rows)
    with pytest.raises(orojenesis.OrojenesisError, match="fields disagree"):
        orojenesis.parse_multi_mapping_records(invalid, word_bytes=2)

    empty = tmp_path / "empty.csv"
    empty.write_text("short,row\n", encoding="utf-8")
    with pytest.raises(orojenesis.OrojenesisError, match="no mapping records"):
        orojenesis.parse_multi_mapping_records(empty, word_bytes=2)
    with pytest.raises(orojenesis.OrojenesisError, match="matrix is incomplete"):
        orojenesis.compose_multi_einsum_curve([[empty]], row_tiles=[1], word_bytes=2)

    left = _write_mapping(tmp_path / "left.csv", output_util=8)
    right = _write_mapping(tmp_path / "right.csv", input_util=16)
    with pytest.raises(orojenesis.OrojenesisError, match="no compatible"):
        orojenesis.compose_multi_einsum_curve(
            [[left], [right]], row_tiles=[1], word_bytes=2
        )


def test_serialized_curve_validation(tmp_path):
    path = tmp_path / "curve.csv"
    path.write_text("bad\n", encoding="utf-8")
    with pytest.raises(orojenesis.OrojenesisError, match="no valid points"):
        orojenesis.parse_multi_einsum_curve(path, word_bytes=2)
    path.write_text("1,1,1,[1],1\n", encoding="utf-8")
    with pytest.raises(orojenesis.OrojenesisError, match="invalid"):
        orojenesis.parse_multi_einsum_curve(path, word_bytes=2)


def test_batched_layout_and_fanout_region_discovery():
    batched = _linear_layers(batched=True)
    batched_regions = orojenesis.find_multi_einsum_regions(batched)
    assert batched_regions[0]["kind"] == "broadcast_batch_linear_matmul"
    assert (
        orojenesis.multi_einsum_region_problem(batched_regions[0]) == batched_regions[0]
    )

    layout = {
        "x": _start("x"),
        "w0": _start("w0"),
        "w1": _start("w1"),
        "mm0": _matmul("mm0", "x", "w0", "hidden", m=2, k=3, n=4),
        "transpose": _view("hidden", "transposed"),
        "mm1": _matmul("mm1", "transposed", "w1", "result", m=4, k=2, n=5),
    }
    layout_region = orojenesis.find_multi_einsum_regions(layout)[0]
    assert layout_region["kind"] == "linear_matmul_with_axis_maps"
    assert layout_region["edges"][0]["axis_map"] == [1, 0]
    assert layout_region["physical_paths"] == [["mm0", "transpose", "mm1"]]

    fanout = {
        "x": _start("x"),
        "w0": _start("w0"),
        "w1": _start("w1"),
        "w2": _start("w2"),
        "mm0": _matmul("mm0", "x", "w0", "hidden"),
        "mm1": _matmul("mm1", "hidden", "w1", "left", k=4, n=5),
        "mm2": _matmul("mm2", "hidden", "w2", "right", k=4, n=6),
    }
    fanout_region = orojenesis.find_multi_einsum_regions(fanout)[0]
    assert fanout_region["kind"] == "matmul_fanout_tree"
    assert fanout_region["leaves"] == ["mm1", "mm2"]


def test_region_roles_and_curve_composition(tmp_path):
    region = orojenesis.find_multi_einsum_regions(_linear_layers(batched=True))[0]
    assert orojenesis.multi_einsum_region_mapper_role(region, "mm0") == "first"
    assert orojenesis.multi_einsum_region_mapper_role(region, "mm1") == "second_last"
    with pytest.raises(orojenesis.OrojenesisError, match="unknown"):
        orojenesis.multi_einsum_region_mapper_role(region, "missing")

    first = _write_mapping(
        tmp_path / "first.csv", input_util=24, output_util=24, mapping="a"
    )
    second = _write_mapping(
        tmp_path / "second.csv", input_util=24, output_util=24, mapping="b"
    )
    curve = orojenesis.compose_multi_einsum_region_curve(
        region,
        {"mm0": [first], "mm1": [second]},
        row_tiles_by_node={"mm0": [6], "mm1": [6]},
        word_bytes=2,
    )
    assert curve[0]["mappings"] == ["a", "b"]

    serialized = tmp_path / "region.csv"
    with serialized.open("w", newline="") as handle:
        csv.writer(handle).writerow(
            [128, 4.0, curve[0]["dram_accesses_words"], '["a","b"]']
        )
    assert (
        orojenesis.parse_multi_einsum_region_curve(serialized, word_bytes=2)[0][
            "dram_bytes"
        ]
        > 0
    )


def test_region_validation_errors(tmp_path):
    region = orojenesis.find_multi_einsum_regions(_linear_layers(batched=True))[0]
    mutations = [
        lambda item: item.update(schema_version=0),
        lambda item: item.update(composition="wrong"),
        lambda item: item.update(schedule=list(reversed(item["schedule"]))),
        lambda item: item["nodes"][0].update(m=0),
        lambda item: item["edges"][0].update(axis_map=[2, 0]),
        lambda item: item.update(roots=[]),
    ]
    for mutation in mutations:
        candidate = deepcopy(region)
        mutation(candidate)
        with pytest.raises(orojenesis.OrojenesisError):
            orojenesis.multi_einsum_region_problem(candidate)

    mapping = _write_mapping(tmp_path / "mapping.csv", input_util=5, output_util=5)
    with pytest.raises(orojenesis.OrojenesisError, match="rectangular"):
        orojenesis.compose_multi_einsum_region_curve(
            region,
            {"mm0": [mapping], "mm1": [mapping]},
            row_tiles_by_node={"mm0": [2], "mm1": [2]},
            word_bytes=2,
        )
    with pytest.raises(orojenesis.OrojenesisError, match="width"):
        orojenesis.compose_multi_einsum_region_curve(
            region, {}, row_tiles_by_node={}, word_bytes=0
        )

    serialized = tmp_path / "serialized.csv"
    serialized.write_text("1,1,1,[1]\n", encoding="utf-8")
    with pytest.raises(orojenesis.OrojenesisError, match="invalid"):
        orojenesis.parse_multi_einsum_region_curve(serialized, word_bytes=2)
    serialized.write_text("bad\n", encoding="utf-8")
    with pytest.raises(orojenesis.OrojenesisError, match="no points"):
        orojenesis.parse_multi_einsum_region_curve(serialized, word_bytes=2)

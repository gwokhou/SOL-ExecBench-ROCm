from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import yaml

from solar.common.constants import SAFE_ENV_VARS
from solar.common.types import TensorShapes
from solar.common.utils import (
    FlowDict,
    FlowList,
    NoAliasDumper,
    convert_numpy_types,
    ensure_directory,
    flowify,
    format_number,
    get_file_prefix,
    load_einsum_graph_to_networkx,
    load_module_from_file,
    merge_dicts,
    parse_dim_tokens,
    parse_einsum_equation,
    parse_kernel_ids,
    setup_safe_environment,
    validate_dim_tokens,
    validate_einsum_ranks_match_shapes,
    validate_shapes,
    validate_tensor_names_match_shapes,
)
from solar.einsum.ops.base import (
    AFOperand,
    AFOp,
    EinsumOp,
    EinsumOperand,
    compute_cost_from_equation,
)
from solar.einsum.ops.matmul_ops import MatmulHandler


@pytest.mark.parametrize(
    ("number", "formatted"),
    (
        (999, "999"),
        (1_500, "1.50K"),
        (2_500_000, "2.50M"),
        (3_500_000_000, "3.50B"),
        (4_500_000_000_000, "4.50T"),
    ),
)
def test_format_number_uses_expected_magnitude(
    number: int,
    formatted: str,
) -> None:
    assert format_number(number) == formatted


def test_filesystem_and_environment_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    for variable in SAFE_ENV_VARS:
        monkeypatch.delenv(variable, raising=False)

    setup_safe_environment()
    directory = ensure_directory(tmp_path / "nested" / "output")
    module_path = tmp_path / "plugin.py"
    module_path.write_text("VALUE = 42\n", encoding="utf-8")

    assert all(os.environ[key] == value for key, value in SAFE_ENV_VARS.items())
    assert "Safe environment configured" in capsys.readouterr().out
    assert directory.is_dir()
    assert load_module_from_file(module_path).VALUE == 42
    assert get_file_prefix("17_square_matrix.py") == "17"


def test_load_module_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Module file not found"):
        load_module_from_file(tmp_path / "missing.py")


def test_kernel_filter_and_dictionary_merge() -> None:
    files = [Path("1_alpha.py"), Path("2_beta.py"), Path("10_gamma.py")]

    assert parse_kernel_ids(None, files) is files
    assert parse_kernel_ids([2, 10], files) == files[1:]
    assert merge_dicts(
        {"nested": {"keep": 1, "replace": 2}},
        {"nested": {"replace": 3, "add": 4}},
    ) == {"nested": {"keep": 1, "replace": 3, "add": 4}}
    assert merge_dicts({"nested": {"old": 1}}, {"nested": {"new": 2}}, deep=False) == {
        "nested": {"new": 2}
    }


@pytest.mark.parametrize(
    "shapes",
    (
        {},
        {"x": []},
        {"x": "2x3"},
        {"x": [2, 0]},
        {"x": [2, 3.0]},
    ),
)
def test_validate_shapes_rejects_invalid_shapes(shapes: object) -> None:
    assert not validate_shapes(cast(dict[str, list[int]], shapes))


def test_validate_shapes_accepts_positive_integer_lists_and_tuples() -> None:
    shapes = cast(dict[str, list[int]], {"x": [2, 3], "y": (4,)})
    assert validate_shapes(shapes)


def test_convert_numpy_types_recurses_through_containers() -> None:
    value = {
        "integer": np.int64(7),
        "floating": np.float32(1.5),
        "array": np.array([1, 2]),
        "tuple": (np.int32(3), "unchanged"),
    }

    assert convert_numpy_types(value) == {
        "integer": 7,
        "floating": pytest.approx(1.5),
        "array": [1, 2],
        "tuple": (3, "unchanged"),
    }


def test_dimension_token_parsing_handles_numbered_and_compound_ranks() -> None:
    assert parse_dim_tokens("") == []
    assert parse_dim_tokens("A1-b2(P+(R))") == ["A1", "B2", "(P+(R))"]
    assert parse_dim_tokens("AA", validate=False) == ["A", "A"]

    with pytest.raises(ValueError, match="Repeated rank"):
        parse_dim_tokens("A0A0")


def test_dimension_token_validation_reports_duplicates() -> None:
    assert validate_dim_tokens(["A", "B"])
    assert not validate_dim_tokens(["A", "A"])

    with pytest.raises(ValueError, match="Repeated rank"):
        validate_dim_tokens(["A", "A"], raise_on_error=True)


@pytest.mark.parametrize("equation", ("", "AB,BC", "AB->AC->BC"))
def test_parse_einsum_equation_rejects_malformed_equations(equation: str) -> None:
    assert parse_einsum_equation(equation) == ([], [])


def test_parse_einsum_equation_handles_sources_and_inputs() -> None:
    assert parse_einsum_equation("->A1B1") == ([], ["A1", "B1"])
    assert parse_einsum_equation(" AB, B1C -> AC ") == (
        [["A", "B"], ["B1", "C"]],
        ["A", "C"],
    )


def test_einsum_shape_validation_reports_input_and_output_mismatches() -> None:
    valid, message = validate_einsum_ranks_match_shapes(
        "AB,BC->AC",
        {"inputs": [[2, 3], [3, 4]], "outputs": [[2, 4]]},
    )
    assert valid
    assert message == ""

    valid, message = validate_einsum_ranks_match_shapes(
        "AB,BC->AC",
        cast(
            dict[str, list[list[int]]],
            {"inputs": [[2], None], "outputs": [[2, 4, 5]]},
        ),
    )
    assert not valid
    assert "input operand 0" in message
    assert "output has 2 dims" in message

    assert validate_einsum_ranks_match_shapes("invalid", {}) == (True, "")


def test_tensor_name_validation_reports_both_count_mismatches() -> None:
    valid, message = validate_tensor_names_match_shapes(
        {"inputs": ["a", "b"], "outputs": []},
        {"inputs": [[1]], "outputs": [[1]]},
    )

    assert not valid
    assert "Input tensor_names has 2 entries" in message
    assert "Output tensor_names has 0 entries" in message
    assert validate_tensor_names_match_shapes(
        {"inputs": ["a"], "outputs": ["b"]},
        {"inputs": [[1]], "outputs": [[1]]},
    ) == (True, "")


def test_graph_loader_only_links_known_nodes() -> None:
    graph = load_einsum_graph_to_networkx(
        {
            "a": {"connections": {"outputs": ["b", "missing"]}, "kind": "source"},
            "b": {"connections": {"inputs": ["a"], "outputs": []}, "kind": "sink"},
        }
    )

    assert set(graph.nodes) == {"a", "b"}
    assert set(graph.edges) == {("a", "b")}
    assert graph.nodes["a"]["kind"] == "source"


def test_flowify_marks_accelforge_inline_structures() -> None:
    value = {
        "projection": {"A": "a"},
        "nested": {"projection": ["B", "C"]},
        "scalar": {"projection": "D"},
        "tensor_accesses": [{"name": "input"}, "sentinel"],
    }
    result = flowify(value)

    assert isinstance(result["projection"], FlowDict)
    assert isinstance(result["nested"]["projection"], FlowList)
    assert result["scalar"]["projection"] == "D"
    assert isinstance(result["tensor_accesses"][0], FlowDict)
    assert flowify([{"projection": ["A"]}])[0]["projection"] == ["A"]
    assert flowify("value") == "value"


def test_yaml_dumper_does_not_emit_aliases() -> None:
    shared = [1, 2]
    dumped = yaml.dump({"a": shared, "b": shared}, Dumper=NoAliasDumper)

    assert "&id" not in dumped
    assert "*id" not in dumped


def test_einsum_operand_operation_and_cost_contracts() -> None:
    input_operand = EinsumOperand("Input", ["M", "K"])
    output_operand = EinsumOperand("Output", ["M", "N"], is_output=True)
    operation = EinsumOp(
        operands=[
            input_operand,
            EinsumOperand("Weight", ["K", "N"]),
            output_operand,
        ],
        equation="MK,KN->MN",
        name="matmul",
    )

    assert input_operand.to_timeloop_dataspace() == {
        "name": "Input",
        "projection": ["M", "K"],
    }
    assert output_operand.to_timeloop_dataspace()["read_write"] == "true"
    assert operation.input_operands == operation.operands[:2]
    assert operation.output_operands == [output_operand]
    assert operation.to_torch_einsum() == ("torch.einsum('MK,KN->MN', Input, Weight)")
    assert operation.to_torch_einsum(["x", "weight"]) == (
        "torch.einsum('MK,KN->MN', x, weight)"
    )
    assert (
        operation.get_compute_cost(
            TensorShapes(inputs=[[2, 3], [3, 4]], outputs=[[2, 4]])
        )
        == 24
    )

    with pytest.raises(ValueError, match="Number of tensor names"):
        operation.to_torch_einsum(["only_one"])


def test_compute_cost_supports_compound_ranks_and_malformed_equations() -> None:
    assert compute_cost_from_equation("", TensorShapes()) == 0
    assert compute_cost_from_equation("AB", TensorShapes()) == 0
    assert (
        compute_cost_from_equation(
            "BC(P+R)(Q+S),OCRS->BOPQ",
            TensorShapes(
                inputs=[[2, 3, 8, 8], [4, 3, 3, 3]],
                outputs=[[2, 4, 6, 6]],
            ),
        )
        == 7_776
    )


def test_handler_validation_fixes_unary_and_binary_rank_mismatches() -> None:
    handler = MatmulHandler(debug=True)
    unary = EinsumOp(
        [EinsumOperand("Input", ["A"]), EinsumOperand("Output", ["A"], True)],
        "A->A",
        "copy",
        is_real_einsum=False,
        elementwise_op="copy",
        reduction_op="none",
    )
    binary = EinsumOp(
        [
            EinsumOperand("Input", ["A"]),
            EinsumOperand("Input_1", ["A"]),
            EinsumOperand("Output", ["A"], True),
        ],
        "A,A->A",
        "add",
        is_real_einsum=False,
        elementwise_op="add",
        reduction_op="none",
    )

    assert handler.can_handle("MATMUL")
    assert not handler.can_handle("conv2d")
    assert (
        handler._validate_einsum(
            unary, {"inputs": [[2, 3]], "outputs": [[2, 3]]}
        ).equation
        == "AB->AB"
    )
    assert (
        handler._validate_einsum(
            binary, {"inputs": [[2, 3], [3]], "outputs": [[2, 3]]}
        ).equation
        == "AB,B->AB"
    )
    assert handler._validate_einsum(unary, {"inputs": [], "outputs": []}) is unary
    assert handler._validate_einsum(unary, {"inputs": [[2]], "outputs": [[2]]}) is unary


def test_handler_validation_covers_larger_second_input_and_missing_shapes() -> None:
    handler = MatmulHandler()
    operation = EinsumOp(
        [
            EinsumOperand("Input", ["A"]),
            EinsumOperand("Input_1", ["A"]),
            EinsumOperand("Output", ["A"], True),
        ],
        "A,A->A",
        "add",
    )

    corrected = handler._try_fix_einsum_ranks(
        operation,
        {"inputs": [[3], [2, 3]], "outputs": [[2, 3]]},
    )
    assert corrected is not None
    assert corrected.equation == "B,AB->AB"
    assert (
        handler._try_fix_einsum_ranks(operation, {"inputs": [[1]], "outputs": []})
        is None
    )
    assert (
        handler._try_fix_einsum_ranks(
            operation,
            cast(
                dict[str, list[list[int]]],
                {"inputs": [None], "outputs": [[1]]},
            ),
        )
        is None
    )


def test_accelforge_operands_and_operations_serialize_optional_fields() -> None:
    lowered = AFOperand("input", ["M", "N"])
    renamed = AFOperand("output", ["m", "n"], ["Batch", "Channel"], is_output=True)

    assert lowered.to_dict() == {
        "name": "input",
        "projection": ["m", "n"],
    }
    assert renamed.to_dict() == {
        "name": "output",
        "projection": {"BATCH": "m", "CHANNEL": "n"},
        "output": True,
    }
    assert AFOp("copy", [lowered, renamed], is_copy_operation=True).to_dict() == {
        "name": "copy",
        "is_copy_operation": True,
        "tensor_accesses": [lowered.to_dict(), renamed.to_dict()],
    }
    assert AFOp("compute", [lowered]).to_dict() == {
        "name": "compute",
        "tensor_accesses": [lowered.to_dict()],
    }

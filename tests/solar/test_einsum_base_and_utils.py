from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
import yaml

from solar.common.types import TensorShapes
from solar.common.utils import (
    NoAliasDumper,
    ensure_directory,
    parse_dim_tokens,
    parse_einsum_equation,
    validate_dim_tokens,
    validate_einsum_ranks_match_shapes,
    validate_tensor_names_match_shapes,
)
from solar.einsum.ops.base import (
    EinsumOp,
    EinsumOperand,
    compute_cost_from_equation,
)
from solar.einsum.ops.matmul_ops import MatmulHandler


def test_ensure_directory_creates_parents(tmp_path: Path) -> None:
    directory = ensure_directory(tmp_path / "nested" / "output")

    assert directory.is_dir()


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
def test_parse_einsum_equation_rejects_malformed_equations(
    equation: str,
) -> None:
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

    assert (
        operation.get_compute_cost(
            TensorShapes(inputs=[[2, 3], [3, 4]], outputs=[[2, 4]]),
        )
        == 24
    )


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

    assert (
        handler._validate_einsum(
            unary,
            {"inputs": [[2, 3]], "outputs": [[2, 3]]},
        ).equation
        == "AB->AB"
    )
    assert (
        handler._validate_einsum(
            binary,
            {"inputs": [[2, 3], [3]], "outputs": [[2, 3]]},
        ).equation
        == "AB,B->AB"
    )
    assert (
        handler._validate_einsum(unary, {"inputs": [], "outputs": []}) is unary
    )
    assert (
        handler._validate_einsum(unary, {"inputs": [[2]], "outputs": [[2]]})
        is unary
    )


def test_handler_validation_covers_larger_second_input_and_missing_shapes() -> (
    None
):
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
        handler._try_fix_einsum_ranks(
            operation,
            {"inputs": [[1]], "outputs": []},
        )
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

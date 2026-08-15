from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from solar.ir.extended_einsum.operations.analyzer import EinsumAnalyzer
from solar.types import TensorShapes


@dataclass(frozen=True, slots=True, kw_only=True)
class OperationCase:
    name: str
    inputs: list[list[int]]
    equation: str
    outputs: list[list[int]] | None = None
    kwargs: dict[str, Any] | None = None


MATMUL_CASES = (
    OperationCase(name="matmul", inputs=[[8], [8]], equation="K,K->"),
    OperationCase(name="matmul", inputs=[[8], [8, 4]], equation="K,KN->N"),
    OperationCase(name="matmul", inputs=[[3, 8], [8]], equation="MK,K->M"),
    OperationCase(
        name="matmul", inputs=[[8], [2, 3, 8, 4]], equation="K,B0B1KN->B0B1N"
    ),
    OperationCase(name="matmul", inputs=[[3, 8], [8, 4]], equation="MK,KN->MN"),
    OperationCase(
        name="matmul", inputs=[[2, 3, 8], [8, 4]], equation="B0MK,KN->B0MN"
    ),
    OperationCase(
        name="matmul", inputs=[[3, 8], [2, 8, 4]], equation="MK,B0KN->B0MN"
    ),
    OperationCase(
        name="matmul",
        inputs=[[5, 3, 8], [2, 5, 8, 4]],
        equation="B1MK,B0B1KN->B0B1MN",
    ),
    OperationCase(
        name="linear", inputs=[[2, 3, 8]], equation="B0B1K,NK->B0B1N"
    ),
    OperationCase(
        name="bmm", inputs=[[2, 3, 8], [2, 8, 4]], equation="BMK,BKN->BMN"
    ),
)


CONV_CASES = (
    OperationCase(
        name="conv1d",
        inputs=[[2, 4, 16], [8, 4, 3]],
        equation="BC(P+R),OCR->BOP",
    ),
    OperationCase(
        name="conv1d",
        inputs=[[2, 4, 16], [4, 1, 3]],
        equation="BO(P+R),OCR->BOP",
        kwargs={
            "module_args": {"groups": 4, "in_channels": 4, "out_channels": 4},
        },
    ),
    OperationCase(
        name="conv1d",
        inputs=[[2, 4, 16], [8, 2, 3]],
        equation="BGI(P+R),GOIR->BGOP",
        kwargs={
            "module_args": {"groups": 2, "in_channels": 4, "out_channels": 8},
        },
    ),
    OperationCase(
        name="conv2d",
        inputs=[[2, 4, 16, 16], [4, 1, 3, 3]],
        equation="BO(P+R)(Q+S),OCRS->BOPQ",
        kwargs={
            "module_args": {"groups": 4, "in_channels": 4, "out_channels": 4},
        },
    ),
    OperationCase(
        name="conv2d",
        inputs=[[2, 4, 16, 16], [8, 2, 3, 3]],
        equation="BGI(P+R)(Q+S),GOIRS->BGOPQ",
        kwargs={
            "module_args": {"groups": 2, "in_channels": 4, "out_channels": 8},
        },
    ),
    OperationCase(
        name="conv3d",
        inputs=[[2, 4, 8, 10, 12], [6, 4, 3, 3, 3]],
        equation="BC(P+T)(Q+R)(U+S),OCTRS->BOPQU",
        kwargs={"stride": (2, 1, 1), "padding": (1, 0, 0)},
    ),
    OperationCase(
        name="convtranspose1d", inputs=[[2, 4, 8]], equation="BCP,CKR->BK(P+R)"
    ),
    OperationCase(
        name="convtranspose2d",
        inputs=[[2, 4, 8, 8], [4, 6, 3, 3]],
        equation="BCPQ,CKRS->BK(P+R)(Q+S)",
    ),
    OperationCase(
        name="convtranspose3d",
        inputs=[[2, 4, 5, 6, 7]],
        equation="BCPQU,CKTRS->BK(P+T)(Q+R)(U+S)",
    ),
)


MISC_CASES = (
    OperationCase(
        name="embedding", inputs=[[2, 3, 4], [100, 16]], equation="ABC,VD->ABCD"
    ),
    OperationCase(name="gru", inputs=[], equation="SBI,GI,GH,BH->SBH"),
    OperationCase(
        name="gru",
        inputs=[[5, 2, 8], [1, 2, 4], [12, 8], [12, 4]],
        equation="SBI,GI,GH,BH->SBH",
    ),
    OperationCase(
        name="lstm", inputs=[[5, 2, 8]], equation="SBI,GI,GH,BH,BH->SBH"
    ),
    OperationCase(
        name="lstm",
        inputs=[[5, 2, 8], [1, 2, 4], [1, 2, 4], [16, 8], [16, 4]],
        equation="SBI,GI,GH,BH,BH->SBH",
    ),
    OperationCase(name="rnn", inputs=[], equation="SBI,HI,HH,BH->SBH"),
    OperationCase(name="cross_entropy", inputs=[[4, 10]], equation="AB,A->"),
    OperationCase(
        name="cross_entropy",
        inputs=[[4, 10], [4]],
        equation="AB,A->A",
        kwargs={"reduction": "none"},
    ),
    OperationCase(
        name="mse_loss",
        inputs=[[2, 3], [2, 3]],
        equation="AB,AB->",
        outputs=[[]],
    ),
    OperationCase(
        name="smooth_l1_loss",
        inputs=[[2, 3], [2, 3]],
        equation="AB,AB->AB",
        outputs=[[2, 3]],
    ),
    OperationCase(name="clone", inputs=[[2, 3, 4]], equation="ABC->ABC"),
)


@pytest.mark.parametrize(
    "case",
    MATMUL_CASES + CONV_CASES + MISC_CASES,
    ids=lambda case: f"{case.name}-{len(case.inputs)}-inputs-{case.equation}",
)
def test_handler_shape_branches_generate_expected_equation(
    case: OperationCase,
) -> None:
    operation = EinsumAnalyzer().get_einsum_op(
        case.name,
        TensorShapes(inputs=case.inputs, outputs=case.outputs or []),
        **(case.kwargs or {}),
    )

    assert operation.equation == case.equation


@pytest.mark.parametrize(
    ("operation", "inputs", "message"),
    (
        ("bmm", [[2, 3, 4]], "Missing Input/Weight"),
        ("conv1d", [[2, 3, 4]], "Missing Input/Weight"),
        ("conv2d", [[2, 3, 4, 4]], "Missing Input/Weight"),
        ("conv3d", [[2, 3, 4, 4, 4]], "Missing Input/Weight"),
        ("embedding", [[2, 3]], "requires 2 inputs"),
        ("cross_entropy", [], "Missing predictions"),
        ("mse_loss", [], "Missing predictions"),
        ("clone", [], "Missing Input"),
        ("scaled_dot_product_attention", [[1, 2, 3, 4]], "requires 3 input"),
        ("flex_attention", [[1, 2, 3, 4]], "requires 3 input"),
        ("multihead_attention", [], "Missing Input"),
    ),
)
def test_handlers_reject_missing_required_shapes(
    operation: str,
    inputs: list[list[int]],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        EinsumAnalyzer().get_einsum_op(
            operation,
            TensorShapes(inputs=inputs),
        )


def test_attention_handlers_cover_fused_and_composite_contracts() -> None:
    shapes = TensorShapes(
        inputs=[[2, 4, 8, 16], [2, 4, 10, 16], [2, 4, 10, 32]],
    )
    flex = EinsumAnalyzer().get_einsum_op("flex_attention", shapes)
    mha = EinsumAnalyzer().get_einsum_op(
        "multi_head_attention_forward",
        TensorShapes(inputs=[[2, 8, 16]]),
    )

    assert flex.equation == "BHQD,BHKD,BHKV->BHQV"
    assert mha.equation == "BSD->BSD"
    assert not mha.is_real_einsum

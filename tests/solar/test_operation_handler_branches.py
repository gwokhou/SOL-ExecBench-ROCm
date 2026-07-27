from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from solar.common.types import TensorShapes
from solar.einsum.analyzer import EinsumAnalyzer
from solar.einsum.ops.attention_ops import ScaledDotProductAttentionHandler


@dataclass(frozen=True)
class OperationCase:
    name: str
    inputs: list[list[int]]
    equation: str
    outputs: list[list[int]] | None = None
    kwargs: dict[str, Any] | None = None


MATMUL_CASES = (
    OperationCase("matmul", [[8], [8]], "K,K->"),
    OperationCase("matmul", [[8], [8, 4]], "K,KN->N"),
    OperationCase("matmul", [[3, 8], [8]], "MK,K->M"),
    OperationCase("matmul", [[8], [2, 3, 8, 4]], "K,B0B1KN->B0B1N"),
    OperationCase("matmul", [[3, 8], [8, 4]], "MK,KN->MN"),
    OperationCase("matmul", [[2, 3, 8], [8, 4]], "B0MK,KN->B0MN"),
    OperationCase("matmul", [[3, 8], [2, 8, 4]], "MK,B0KN->B0MN"),
    OperationCase(
        "matmul",
        [[5, 3, 8], [2, 5, 8, 4]],
        "B1MK,B0B1KN->B0B1MN",
    ),
    OperationCase("linear", [[2, 3, 8]], "B0B1K,NK->B0B1N"),
    OperationCase("bmm", [[2, 3, 8], [2, 8, 4]], "BMK,BKN->BMN"),
)


CONV_CASES = (
    OperationCase("conv1d", [[2, 4, 16], [8, 4, 3]], "BC(P+R),OCR->BOP"),
    OperationCase(
        "conv1d",
        [[2, 4, 16], [4, 1, 3]],
        "BO(P+R),OCR->BOP",
        kwargs={"module_args": {"groups": 4, "in_channels": 4, "out_channels": 4}},
    ),
    OperationCase(
        "conv1d",
        [[2, 4, 16], [8, 2, 3]],
        "BGI(P+R),GOIR->BGOP",
        kwargs={"module_args": {"groups": 2, "in_channels": 4, "out_channels": 8}},
    ),
    OperationCase(
        "conv2d",
        [[2, 4, 16, 16], [4, 1, 3, 3]],
        "BO(P+R)(Q+S),OCRS->BOPQ",
        kwargs={"module_args": {"groups": 4, "in_channels": 4, "out_channels": 4}},
    ),
    OperationCase(
        "conv2d",
        [[2, 4, 16, 16], [8, 2, 3, 3]],
        "BGI(P+R)(Q+S),GOIRS->BGOPQ",
        kwargs={"module_args": {"groups": 2, "in_channels": 4, "out_channels": 8}},
    ),
    OperationCase(
        "conv3d",
        [[2, 4, 8, 10, 12], [6, 4, 3, 3, 3]],
        "BC(P+T)(Q+R)(U+S),OCTRS->BOPQU",
        kwargs={"stride": (2, 1, 1), "padding": (1, 0, 0)},
    ),
    OperationCase("convtranspose1d", [[2, 4, 8]], "BCP,CKR->BK(P+R)"),
    OperationCase(
        "convtranspose2d",
        [[2, 4, 8, 8], [4, 6, 3, 3]],
        "BCPQ,CKRS->BK(P+R)(Q+S)",
    ),
    OperationCase(
        "convtranspose3d",
        [[2, 4, 5, 6, 7]],
        "BCPQU,CKTRS->BK(P+T)(Q+R)(U+S)",
    ),
)


MISC_CASES = (
    OperationCase("embedding", [[2, 3, 4], [100, 16]], "ABC,VD->ABCD"),
    OperationCase("gru", [], "SBI,GI,GH,BH->SBH"),
    OperationCase(
        "gru",
        [[5, 2, 8], [1, 2, 4], [12, 8], [12, 4]],
        "SBI,GI,GH,BH->SBH",
    ),
    OperationCase("lstm", [[5, 2, 8]], "SBI,GI,GH,BH,BH->SBH"),
    OperationCase(
        "lstm",
        [[5, 2, 8], [1, 2, 4], [1, 2, 4], [16, 8], [16, 4]],
        "SBI,GI,GH,BH,BH->SBH",
    ),
    OperationCase("rnn", [], "SBI,HI,HH,BH->SBH"),
    OperationCase("cross_entropy", [[4, 10]], "AB,A->"),
    OperationCase(
        "cross_entropy",
        [[4, 10], [4]],
        "AB,A->A",
        kwargs={"reduction": "none"},
    ),
    OperationCase("mse_loss", [[2, 3], [2, 3]], "AB,AB->", outputs=[[]]),
    OperationCase(
        "smooth_l1_loss",
        [[2, 3], [2, 3]],
        "AB,AB->AB",
        outputs=[[2, 3]],
    ),
    OperationCase("clone", [[2, 3, 4]], "ABC->ABC"),
)


@pytest.mark.parametrize(
    "case",
    MATMUL_CASES + CONV_CASES + MISC_CASES,
    ids=lambda case: f"{case.name}-{case.equation}",
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
        EinsumAnalyzer().get_einsum_op(operation, TensorShapes(inputs=inputs))


def test_attention_handlers_cover_fused_and_composite_contracts() -> None:
    shapes = TensorShapes(inputs=[[2, 4, 8, 16], [2, 4, 10, 16], [2, 4, 10, 32]])
    flex = EinsumAnalyzer().get_einsum_op("flex_attention", shapes)
    mha = EinsumAnalyzer().get_einsum_op(
        "multi_head_attention_forward",
        TensorShapes(inputs=[[2, 8, 16]]),
    )

    assert flex.equation == "BHQD,BHKD,BHKV->BHQV"
    assert mha.equation == "BSD->BSD"
    assert not mha.is_real_einsum


@pytest.mark.parametrize("with_output", [False, True])
def test_sdpa_subgraph_has_reviewable_shapes_and_connections(
    with_output: bool,
) -> None:
    node_data: dict[str, Any] = {
        "input_shapes": [[2, 4, 8, 16], [2, 4, 10, 16], [2, 4, 10, 32]],
    }
    if with_output:
        node_data["output_shapes"] = [[2, 4, 8, 32]]

    handler = ScaledDotProductAttentionHandler()
    subgraph = handler.create_subgraph("attention", node_data)

    assert list(subgraph) == [
        "attention.qk_matmul",
        "attention.scale",
        "attention.softmax",
        "attention.av_matmul",
    ]
    assert subgraph["attention.qk_matmul"]["output_shapes"] == [[2, 4, 8, 10]]
    assert subgraph["attention.scale"]["module_args"] == {"scale_factor": "1/sqrt(16)"}
    assert subgraph["attention.av_matmul"]["output_shapes"] == [[2, 4, 8, 32]]
    assert handler.get_subgraph_input_mapping("attention") == {
        "attention.qk_matmul": [0, 1],
        "attention.av_matmul": [2],
    }


def test_sdpa_subgraph_rejects_missing_inputs() -> None:
    with pytest.raises(ValueError, match="SDPA requires 3 inputs"):
        ScaledDotProductAttentionHandler().create_subgraph(
            "attention", {"input_shapes": [[2, 4, 8, 16]]}
        )

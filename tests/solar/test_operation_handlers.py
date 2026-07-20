from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from solar.common.types import TensorShapes
from solar.einsum.analyzer import EinsumAnalyzer
from solar.einsum.ops.registry import EinsumOpRegistry, get_global_registry


@dataclass(frozen=True)
class HandlerCase:
    operation: str
    inputs: list[list[int]]
    outputs: list[list[int]] = field(default_factory=list)
    kwargs: dict[str, Any] = field(default_factory=dict)
    equation: str = ""
    elementwise_op: str = ""
    reduction_op: str = ""


CASES = (
    HandlerCase("matmul", [[2, 3], [3, 4]], equation="MK,KN->MN"),
    HandlerCase(
        "add",
        [[2, 3], [3]],
        [[2, 3]],
        equation="AB,B->AB",
        elementwise_op="add",
        reduction_op="none",
    ),
    HandlerCase(
        "sum",
        [[2, 3, 4]],
        [[2, 4]],
        {"dims": [1]},
        equation="ABC->AC",
        reduction_op="add",
    ),
    HandlerCase(
        "conv2d",
        [[1, 3, 8, 8], [4, 3, 3, 3]],
        [[1, 4, 6, 6]],
        equation="BC(P+R)(Q+S),OCRS->BOPQ",
    ),
    HandlerCase(
        "scaled_dot_product_attention",
        [[2, 4, 8, 16], [2, 4, 10, 16], [2, 4, 10, 32]],
        [[2, 4, 8, 32]],
        equation="BHQD,BHKD,BHKV->BHQV",
    ),
    HandlerCase(
        "cumsum",
        [[2, 3, 4]],
        [[2, 3, 4]],
        {"dim": 1},
        equation="ABC->ABC",
        reduction_op="add",
    ),
    HandlerCase(
        "poisson_nll_loss",
        [[2, 3]],
        [[]],
        equation="AB->",
        elementwise_op="poisson_nll",
        reduction_op="add",
    ),
    HandlerCase(
        "layer_norm",
        [[2, 3, 4]],
        [[2, 3, 4]],
        equation="ABC->ABC",
        elementwise_op="layer_norm",
        reduction_op="none",
    ),
    HandlerCase(
        "max_pool2d",
        [[2, 3, 8, 8]],
        [[2, 3, 4, 4]],
        {"kernel_size": (2, 2)},
        equation="ABCD->ABCD",
        reduction_op="max",
    ),
    HandlerCase(
        "embedding",
        [[2, 3], [100, 16]],
        [[2, 3, 16]],
        equation="AB,VD->ABD",
        elementwise_op="embedding",
    ),
    HandlerCase(
        "reshape",
        [[2, 3]],
        [[6]],
        equation="AB->R0",
        elementwise_op="copy",
        reduction_op="none",
    ),
    HandlerCase(
        "diag",
        [[3]],
        [[3, 3]],
        equation="A->AB",
        elementwise_op="copy",
        reduction_op="none",
    ),
)


def test_global_registry_loads_every_builtin_handler_family() -> None:
    registry = get_global_registry()

    assert {
        "add",
        "conv2d",
        "cumsum",
        "embedding",
        "layer_norm",
        "matmul",
        "max_pool2d",
        "poisson_nll_loss",
        "reshape",
        "scaled_dot_product_attention",
        "sum",
    } <= set(registry.list_supported_ops())
    assert registry.get_statistics()["total_handlers"] >= 20


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.operation)
def test_builtin_handlers_generate_reviewable_operation_contracts(
    case: HandlerCase,
) -> None:
    operation = EinsumAnalyzer().get_einsum_op(
        case.operation,
        TensorShapes(inputs=case.inputs, outputs=case.outputs),
        **case.kwargs,
    )

    assert operation.equation == case.equation
    if case.elementwise_op:
        assert operation.elementwise_op == case.elementwise_op
    if case.reduction_op:
        assert operation.reduction_op == case.reduction_op


def test_registry_rejects_unknown_operation() -> None:
    registry = EinsumOpRegistry()

    with pytest.raises(ValueError, match="No handler registered"):
        registry.get_einsum_op("unknown", TensorShapes(inputs=[[2]]))

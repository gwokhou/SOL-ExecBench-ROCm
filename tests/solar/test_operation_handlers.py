from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from solar.ir.extended_einsum.operations.analyzer import EinsumAnalyzer
from solar.ir.extended_einsum.operations.handlers.base import (
    EinsumOp,
    EinsumOperand,
    EinsumOpHandler,
)
from solar.ir.extended_einsum.operations.handlers.registry import (
    EinsumOpRegistry,
    EinsumOpRegistryBuilder,
    build_builtin_registry,
    builtin_einsum_registry,
)
from solar.types import TensorShapes


@dataclass(frozen=True, slots=True, kw_only=True)
class HandlerCase:
    operation: str
    inputs: list[list[int]]
    outputs: list[list[int]] = field(default_factory=list)
    kwargs: dict[str, Any] = field(default_factory=dict)
    equation: str = ""
    elementwise_op: str = ""
    reduction_op: str = ""


CASES = (
    HandlerCase(
        operation="matmul", inputs=[[2, 3], [3, 4]], equation="MK,KN->MN"
    ),
    HandlerCase(
        operation="add",
        inputs=[[2, 3], [3]],
        outputs=[[2, 3]],
        equation="AB,B->AB",
        elementwise_op="add",
        reduction_op="none",
    ),
    HandlerCase(
        operation="sum",
        inputs=[[2, 3, 4]],
        outputs=[[2, 4]],
        kwargs={"dims": [1]},
        equation="ABC->AC",
        reduction_op="add",
    ),
    HandlerCase(
        operation="conv2d",
        inputs=[[1, 3, 8, 8], [4, 3, 3, 3]],
        outputs=[[1, 4, 6, 6]],
        equation="BC(P+R)(Q+S),OCRS->BOPQ",
    ),
    HandlerCase(
        operation="scaled_dot_product_attention",
        inputs=[[2, 4, 8, 16], [2, 4, 10, 16], [2, 4, 10, 32]],
        outputs=[[2, 4, 8, 32]],
        equation="BHQD,BHKD,BHKV->BHQV",
    ),
    HandlerCase(
        operation="cumsum",
        inputs=[[2, 3, 4]],
        outputs=[[2, 3, 4]],
        kwargs={"dim": 1},
        equation="ABC->ABC",
        reduction_op="add",
    ),
    HandlerCase(
        operation="poisson_nll_loss",
        inputs=[[2, 3]],
        outputs=[[]],
        equation="AB->",
        elementwise_op="poisson_nll",
        reduction_op="add",
    ),
    HandlerCase(
        operation="layer_norm",
        inputs=[[2, 3, 4]],
        outputs=[[2, 3, 4]],
        equation="ABC->ABC",
        elementwise_op="layer_norm",
        reduction_op="none",
    ),
    HandlerCase(
        operation="max_pool2d",
        inputs=[[2, 3, 8, 8]],
        outputs=[[2, 3, 4, 4]],
        kwargs={"kernel_size": (2, 2)},
        equation="ABCD->ABCD",
        reduction_op="max",
    ),
    HandlerCase(
        operation="embedding",
        inputs=[[2, 3], [100, 16]],
        outputs=[[2, 3, 16]],
        equation="AB,VD->ABD",
        elementwise_op="embedding",
    ),
    HandlerCase(
        operation="reshape",
        inputs=[[2, 3]],
        outputs=[[6]],
        equation="AB->R0",
        elementwise_op="copy",
        reduction_op="none",
    ),
    HandlerCase(
        operation="diag",
        inputs=[[3]],
        outputs=[[3, 3]],
        equation="A->AB",
        elementwise_op="copy",
        reduction_op="none",
    ),
)


class _InjectedHandler(EinsumOpHandler):
    supported_ops = ("injected",)

    def generate_einsum(
        self,
        op_name: str,
        tensor_shapes: TensorShapes,
        **kwargs: Any,
    ) -> EinsumOp:
        del tensor_shapes, kwargs
        return EinsumOp(
            operands=[
                EinsumOperand(name="Input", dims=["A"]),
                EinsumOperand(name="Output", dims=["A"], is_output=True),
            ],
            equation="A->A",
            name=op_name,
        )


def test_builtin_registry_loads_every_handler_family() -> None:
    registry = builtin_einsum_registry()

    assert all(
        registry.has_handler(operation)
        for operation in (
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
        )
    )


def test_builtin_registry_builds_independent_instances() -> None:
    first = build_builtin_registry()
    second = build_builtin_registry()

    assert first is not second
    assert not second.has_handler("injected")


def test_registry_builder_builds_an_immutable_snapshot() -> None:
    builder = EinsumOpRegistryBuilder()
    registry = builder.register_handler(_InjectedHandler).build()

    assert registry.has_handler("injected")
    mutable_handlers: Any = registry.handlers
    with pytest.raises(TypeError):
        mutable_handlers["other"] = registry.handlers["injected"]


def test_einsum_operand_constructor_is_keyword_only() -> None:
    constructor: Any = EinsumOperand
    with pytest.raises(TypeError):
        constructor("Input", ["M"])


def test_registry_requires_explicit_handler_replacement() -> None:
    registry = EinsumOpRegistryBuilder()
    registry.register_handler(_InjectedHandler)

    with pytest.raises(ValueError, match="would replace registered handlers"):
        registry.register_handler(_InjectedHandler)

    registry.register_handler(
        _InjectedHandler,
        replace_ops=frozenset({"injected"}),
    )


def test_analyzer_uses_injected_registry_without_global_fallback() -> None:
    registry = EinsumOpRegistryBuilder()
    registry.register_handler(_InjectedHandler)
    analyzer = EinsumAnalyzer(registry=registry.build())

    operation = analyzer.get_einsum_op(
        "injected",
        TensorShapes(inputs=[[2]], outputs=[[2]]),
    )

    assert operation.equation == "A->A"
    with pytest.raises(ValueError, match="Unsupported operation"):
        analyzer.get_einsum_op("matmul", TensorShapes(inputs=[[2], [2]]))


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
    registry = EinsumOpRegistry({})

    with pytest.raises(ValueError, match="No handler registered"):
        registry.get_einsum_op("unknown", TensorShapes(inputs=[[2]]))

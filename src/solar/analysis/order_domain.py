# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Recognize a pure, ordered two-dimensional matrix chain in validated IR."""

from __future__ import annotations

from dataclasses import dataclass

from solar.analysis.order_trees import (
    MATMUL_TOKEN,
    MAX_ORDER_LEAVES,
    OrderTree,
    tree_cost,
)
from solar.errors import SolarError
from solar.ir.contracts import (
    layer_operation,
    operation_attributes,
    operation_operands,
)
from solar.types import NodeDict

ORDER_DTYPES = frozenset(
    {"torch.float16", "torch.bfloat16", "torch.float32", "torch.float64"}
)
MATRIX_TARGETS = frozenset({"mm", "matmul"})


class OrderDomainError(SolarError):
    """The valid source graph lies outside the reviewed reassociation domain."""

    reason_code = "contraction_order_domain_unsupported"


@dataclass(frozen=True, slots=True, kw_only=True)
class MatrixChain:
    """Source operand order, adjacent index sizes and original association."""

    source_indices: tuple[int, ...]
    input_names: tuple[str, ...]
    dimensions: tuple[int, ...]
    dtype: str
    original: OrderTree
    output_name: str


def _matrix_operation(layer: NodeDict) -> None:
    operation = layer_operation(layer)
    effects = operation.get("effects", {})
    if any(
        effects.get(key)
        for key in ("mutates", "aliases", "atomic", "opaque_library_call")
    ):
        raise OrderDomainError("matrix chain contains effects or aliases")
    if operation_attributes(operation) or operation_operands(operation) != [
        {"tensor": 0},
        {"tensor": 1},
    ]:
        raise OrderDomainError(
            "matrix call has noncanonical operands or attributes"
        )
    target = operation.get("target")
    if target == "einsum":
        equation = operation.get("equation", "")
        if not _matrix_equation(equation):
            raise OrderDomainError(
                "only ordered matrix-index contractions are supported"
            )
    elif target not in MATRIX_TARGETS:
        raise OrderDomainError("source contains a non-matrix operation")


def _matrix_equation(equation: str) -> bool:
    # IR already validated the equation. This is the smaller search domain.
    if len(equation) != 9 or equation[2] != "," or equation[5:7] != "->":
        return False
    left, right, output = equation[:2], equation[3:5], equation[7:]
    return (
        len(set(left + right)) == 3
        and left[1] == right[0]
        and output == left[0] + right[1]
    )


class _ChainReader:
    def __init__(self, graph: NodeDict) -> None:
        self.layers = graph["layers"]
        if len(self.layers) > 2 * MAX_ORDER_LEAVES - 1:
            raise OrderDomainError(
                "source exceeds the matrix chain node budget"
            )
        self.producers = {
            name: key
            for key, layer in self.layers.items()
            for name in layer["tensor_names"]["outputs"]
        }
        self.visited: set[str] = set()
        self.inputs: list[tuple[int, str, tuple[int, int], str]] = []

    def visit(self, name: str) -> tuple[OrderTree, tuple[int, int], str]:
        key = self.producers[name]
        if key in self.visited:
            raise OrderDomainError(
                "shared or repeated operands are outside the chain domain"
            )
        self.visited.add(key)
        layer = self.layers[key]
        shapes = layer["tensor_shapes"]["outputs"]
        dtypes = layer["tensor_dtypes"]["outputs"]
        if len(shapes) != 1 or len(shapes[0]) != 2 or len(dtypes) != 1:
            raise OrderDomainError(
                "search requires one rank-two output per node"
            )
        shape = tuple(shapes[0])
        if (
            any(type(d) is not int or d < 0 for d in shape)
            or dtypes[0] not in ORDER_DTYPES
        ):
            raise OrderDomainError(
                "search requires static shapes and floating matrix dtypes"
            )
        if layer["type"] == "start":
            index = layer["source_input_index"]
            leaf = len(self.inputs)
            self.inputs.append((index, name, shape, dtypes[0]))
            return (leaf,), shape, dtypes[0]
        _matrix_operation(layer)
        names = layer["tensor_names"]["inputs"]
        if len(names) != 2:
            raise OrderDomainError("matrix contraction requires two operands")
        left, ls, ld = self.visit(names[0])
        right, rs, rd = self.visit(names[1])
        if (
            ls[1] != rs[0]
            or shape != (ls[0], rs[1])
            or not ld == rd == dtypes[0]
        ):
            raise ValueError(
                "validated matrix graph has inconsistent shape or dtype"
            )
        return (*left, *right, MATMUL_TOKEN), shape, dtypes[0]


def matrix_chain(graph: NodeDict) -> MatrixChain:
    """Read an already validated graph; reject unsupported domain, not bad IR."""
    outputs = graph["outputs"]
    if len(outputs) != 1:
        raise OrderDomainError("search requires exactly one matrix output")
    reader = _ChainReader(graph)
    original, _, dtype = reader.visit(outputs[0])
    if reader.visited != set(reader.layers):
        raise OrderDomainError("source has operations outside the output chain")
    if not 2 <= len(reader.inputs) <= MAX_ORDER_LEAVES:
        raise OrderDomainError(
            "search requires a bounded chain of at least two matrices"
        )
    indices = tuple(item[0] for item in reader.inputs)
    if len(set(indices)) != len(indices) or set(indices) != set(
        graph["source_input_indices"]
    ):
        raise OrderDomainError(
            "source operands must have distinct exact bindings"
        )
    dimensions = (
        reader.inputs[0][2][0],
        *(item[2][1] for item in reader.inputs),
    )
    tree_cost(original, dimensions)
    return MatrixChain(
        source_indices=indices,
        input_names=tuple(item[1] for item in reader.inputs),
        dimensions=dimensions,
        dtype=dtype,
        original=original,
        output_name=outputs[0],
    )

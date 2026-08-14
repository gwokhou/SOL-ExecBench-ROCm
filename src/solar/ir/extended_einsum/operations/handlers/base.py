# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Base classes for einsum operation handlers.

This module defines the core data structures and abstract base class
for all einsum operation handlers.
"""

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Optional

from solar.ir.extended_einsum.equations import (
    validate_einsum_ranks_match_shapes,
)
from solar.types import DynamicValue, TensorShape, TensorShapes

logger = logging.getLogger(__name__)


@dataclass
class EinsumOperand:
    """Represents an operand in an einsum operation."""

    name: str
    dims: list[str]
    is_output: bool = False
    stride: dict[str, int] | None = None
    dilation: dict[str, int] | None = None


@dataclass
class EinsumOp:
    """Represents an einsum operation.

    The extended-einsum representation supports elementwise and reduction
    operations beyond the standard multiply-add semantics:

    - elementwise_op: The operation applied element-wise (default: 'mul')
      Examples: 'mul' for matmul, 'add' for element-wise add, 'max' for max pooling
    - reduction_op: The operation used to reduce/aggregate (default: 'add')
      Examples: 'add' for sum, 'max' for max reduction, 'none' for no reduction
    - is_real_einsum: True if this is a standard tensor contraction (mul+add)
    - is_einsum_supportable: whether extended-einsum can express the operation

    Standard einsum (matmul): elementwise_op='mul', reduction_op='add'
    Element-wise add:        elementwise_op='add', reduction_op='none'
    Max pooling:             elementwise_op='copy', reduction_op='max'
    """

    operands: list[EinsumOperand]
    equation: str
    name: str
    is_real_einsum: bool = True
    elementwise_op: str = (
        "mul"  # 'mul', 'add', 'sub', 'div', 'max', 'min', 'copy'
    )
    reduction_op: str = "add"  # 'add', 'max', 'min', 'mul', 'none'
    is_einsum_supportable: bool = (
        True  # Can extended-einsum express this operation?
    )

    def get_compute_cost(self, tensor_shapes: TensorShapes) -> int:
        """Calculate compute cost from einsum rank dimensions.

        Collects unique rank dimension sizes from ALL operands (input + output).
        Compound dims like 'P+R' are split into atoms and resolved from other
        operands. No op-specific special cases — purely driven by einsum equation.

        Total cost = product of all unique resolved rank dimension sizes.
        """
        return compute_cost_from_equation(self.equation, tensor_shapes)


def _parse_dim_atoms(dim: str) -> list[str]:
    """Parse a possibly compound dim into atomic rank names.

    'P+R' -> ['P', 'R']
    'B'   -> ['B']
    'P+R0' -> ['P', 'R0']
    """
    return [d.strip() for d in re.split(r"[+\-]", dim) if d.strip()]


def _parse_equation_operand(operand: str) -> list[str]:
    """Parse one einsum operand into rank tokens.

    Supports single-letter ranks with optional digits and parenthesized
    compound ranks, e.g. ``BGI(P+R)`` -> ``["B", "G", "I", "P+R"]``.
    """
    tokens: list[str] = []
    i = 0
    while i < len(operand):
        if operand[i] == "(":
            j = operand.index(")", i)
            tokens.append(operand[i + 1 : j])
            i = j + 1
        elif operand[i].isalpha():
            token = operand[i]
            i += 1
            while i < len(operand) and operand[i].isdigit():
                token += operand[i]
                i += 1
            tokens.append(token)
        else:
            i += 1
    return tokens


def compute_cost_from_equation(
    equation: str,
    tensor_shapes: TensorShapes,
) -> int:
    """Calculate compute cost from an einsum equation and tensor shapes.

    The cost model is the product of every unique rank used by the equation.
    Compound ranks like ``P+R`` are split into atoms; kernel atoms such as
    ``R`` are resolved from concrete input operands, and output-position atoms
    such as ``P`` are resolved from the output shapes.
    """
    if not equation or "->" not in equation:
        return 0

    lhs, rhs = equation.split("->", 1)
    input_tokens = [
        _parse_equation_operand(operand) for operand in lhs.split(",")
    ]
    output_tokens = _parse_equation_operand(rhs)

    all_ranks: dict[str, int | None] = {}
    for tokens in input_tokens:
        for token in tokens:
            for atom in _parse_dim_atoms(token):
                all_ranks.setdefault(atom, None)
    for token in output_tokens:
        for atom in _parse_dim_atoms(token):
            all_ranks.setdefault(atom, None)

    def _resolve(
        tokens_by_operand: list[list[str]],
        shapes: list[TensorShape],
    ) -> None:
        for idx, tokens in enumerate(tokens_by_operand):
            if idx >= len(shapes):
                break
            shape = shapes[idx]
            for dim_offset, token in enumerate(tokens):
                atoms = _parse_dim_atoms(token)
                if len(atoms) == 1 and dim_offset < len(shape):
                    atom = atoms[0]
                    if all_ranks.get(atom) is None:
                        all_ranks[atom] = int(shape[dim_offset])

    _resolve(input_tokens, tensor_shapes.inputs)
    _resolve([output_tokens], tensor_shapes.outputs)

    total_ops = 1
    for value in all_ranks.values():
        if value is not None and value > 0:
            total_ops *= value
    return int(total_ops)


class EinsumOpHandler(ABC):
    """Abstract base class for einsum operation handlers.

    Each handler is responsible for converting one or more related operation
    types to einsum notation. Handlers receive TensorShapes (positional)
    and should access inputs/outputs by index, not by name.
    """

    supported_ops: ClassVar[tuple[str, ...]] = ()

    @abstractmethod
    def generate_einsum(
        self,
        op_name: str,
        tensor_shapes: TensorShapes,
        **kwargs: DynamicValue,
    ) -> EinsumOp:
        """Generate an einsum operation for the given operation.

        Args:
            op_name: Normalized operation name.
            tensor_shapes: Positional input/output shapes.
            **kwargs: Additional operation-specific parameters.

        Returns:
            EinsumOp representing the operation.

        """

    def _validate_einsum(
        self,
        einsum_op: "EinsumOp",
        tensor_shapes: dict[str, list[list[int]]],
    ) -> "EinsumOp":
        """Validate that einsum ranks match tensor shapes.

        If validation fails, logs a warning and attempts to fix the equation
        by regenerating it based on actual shapes.

        Args:
            einsum_op: The generated EinsumOp to validate.
            tensor_shapes: Dictionary with "inputs" and "outputs" keys containing shape lists.
                          Format: {"inputs": [[shape1], [shape2]], "outputs": [[output_shape]]}

        Returns:
            The validated (and possibly corrected) EinsumOp.

        """
        is_valid, error_msg = validate_einsum_ranks_match_shapes(
            einsum_op.equation,
            tensor_shapes,
        )

        if not is_valid:
            logger.warning(
                "Einsum rank mismatch for %s: %s. Equation: %s, "
                "tensor_shapes: %s",
                einsum_op.name,
                error_msg,
                einsum_op.equation,
                tensor_shapes,
            )
            # Try to fix by regenerating equation from shapes
            corrected_op = self._try_fix_einsum_ranks(einsum_op, tensor_shapes)
            if corrected_op is not None:
                return corrected_op

        return einsum_op

    def _try_fix_einsum_ranks(
        self,
        einsum_op: "EinsumOp",
        tensor_shapes: dict[str, list[list[int]]],
    ) -> Optional["EinsumOp"]:
        """Best-effort regenerate an equation from observed tensor ranks."""
        import string

        # Get actual shapes from tensor_shapes
        input_shapes = tensor_shapes.get("inputs", [])
        output_shapes = tensor_shapes.get("outputs", [])

        if not input_shapes or not output_shapes:
            return None

        input_shape = input_shapes[0] if input_shapes else None
        input_1_shape = input_shapes[1] if len(input_shapes) > 1 else None
        output_shape = output_shapes[0] if output_shapes else None

        if input_shape is None or output_shape is None:
            return None

        input_rank = len(input_shape)
        output_rank = len(output_shape)

        # Generate labels based on actual ranks
        input_labels = string.ascii_uppercase[:input_rank]
        output_labels = string.ascii_uppercase[:output_rank]

        # For binary ops, handle second input
        if input_1_shape is not None:
            input_1_rank = len(input_1_shape)

            # Handle broadcasting: use output labels for the larger tensor
            if input_1_rank < input_rank:
                # A smaller second input uses the output-label suffix.
                input_1_labels = (
                    output_labels[-input_1_rank:] if input_1_rank > 0 else ""
                )
            elif input_1_rank > input_rank:
                # First input is smaller, use suffix of output labels
                input_labels = (
                    output_labels[-input_rank:] if input_rank > 0 else ""
                )
                input_1_labels = output_labels
            else:
                input_1_labels = input_labels

            new_equation = f"{input_labels},{input_1_labels}->{output_labels}"

            # Update operands
            new_operands = [
                EinsumOperand("Input", list(input_labels), is_output=False),
                EinsumOperand("Input_1", list(input_1_labels), is_output=False),
                EinsumOperand("Output", list(output_labels), is_output=True),
            ]
        else:
            new_equation = f"{input_labels}->{output_labels}"
            new_operands = [
                EinsumOperand("Input", list(input_labels), is_output=False),
                EinsumOperand("Output", list(output_labels), is_output=True),
            ]

        logger.info(
            "Fixed einsum equation: %s -> %s",
            einsum_op.equation,
            new_equation,
        )

        return EinsumOp(
            operands=new_operands,
            equation=new_equation,
            name=einsum_op.name,
            is_real_einsum=einsum_op.is_real_einsum,
            elementwise_op=einsum_op.elementwise_op,
            reduction_op=einsum_op.reduction_op,
            is_einsum_supportable=einsum_op.is_einsum_supportable,
        )


__all__ = [
    "EinsumOp",
    "EinsumOpHandler",
    "EinsumOperand",
    "compute_cost_from_equation",
]

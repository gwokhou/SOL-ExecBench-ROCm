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

"""Core einsum analyzer for converting operations to einsum notation.

This module provides the EinsumAnalyzer class which uses the registered
operation handlers to convert PyTorch operations to einsum notation.
"""

import re
from collections.abc import Callable
from typing import Any, Dict, List, Optional, Tuple

from solar.common.types import TensorShape, TensorShapes
from solar.einsum.ops.base import (
    EinsumOp,
    compute_cost_from_equation,
)
from solar.einsum.ops.registry import get_global_registry


_CONVOLUTION_ALIASES = (
    (("convtranspose1d", "conv_transpose1d"), "convtranspose1d"),
    (("convtranspose2d", "conv_transpose2d"), "convtranspose2d"),
    (("convtranspose3d", "conv_transpose3d"), "convtranspose3d"),
    (("conv1d",), "conv1d"),
    (("conv2d",), "conv2d"),
    (("conv3d",), "conv3d"),
)
_LOSS_ALIASES = (
    (("poisson_nll_loss",), "poisson_nll_loss"),
    (("smooth_l1_loss",), "smooth_l1_loss"),
    (("binary_cross_entropy", "bce_loss"), "bce_loss"),
    (("cosine_embedding_loss",), "cosine_embedding_loss"),
    (("hinge_embedding_loss",), "hinge_embedding_loss"),
    (("margin_ranking_loss",), "margin_ranking_loss"),
    (("triplet_margin_loss",), "triplet_margin_loss"),
    (("cross_entropy",), "cross_entropy"),
    (("kl_div",), "kl_div"),
    (("nll_loss",), "nll_loss"),
    (("mse_loss",), "mse_loss"),
    (("l1_loss",), "l1_loss"),
    (("huber_loss",), "huber_loss"),
    (("ctc_loss",), "ctc_loss"),
)
_ACTIVATION_ALIASES = (
    (("hardsigmoid",), "hardsigmoid"),
    (("hardswish",), "hardswish"),
    (("hardtanh",), "hardtanh"),
    (("leaky_relu",), "leaky_relu"),
    (("prelu",), "prelu"),
    (("rrelu",), "rrelu"),
    (("relu",), "relu"),
    (("sigmoid",), "sigmoid"),
    (("tanh",), "tanh"),
    (("gelu",), "gelu"),
    (("selu",), "selu"),
    (("celu",), "celu"),
    (("elu",), "elu"),
    (("mish",), "mish"),
    (("silu",), "silu"),
    (("log_softmax",), "log_softmax"),
    (("softmax",), "softmax"),
    (("softplus",), "softplus"),
    (("softsign",), "softsign"),
    (("clamp",), "clamp"),
)
_INDEXED_WRITES = {"index_add", "index_copy", "index_put", "scatter", "scatter_add"}
_MATH_OPERATIONS = {
    "abs",
    "neg",
    "exp",
    "log",
    "log2",
    "log10",
    "sqrt",
    "rsqrt",
    "sin",
    "cos",
    "tan",
}


def _first_contained_alias(
    operation: str,
    aliases: tuple[tuple[tuple[str, ...], str], ...],
) -> str | None:
    for needles, canonical in aliases:
        if any(needle in operation for needle in needles):
            return canonical
    return None


def _matches_qualified_alias(operation: str, aliases: set[str]) -> bool:
    return operation in aliases or any(
        operation.endswith(f".{alias}") for alias in aliases
    )


def _strip_operation_syntax(operation: str) -> str:
    normalized = operation.lower()
    for prefix in ("torch.nn.", "torch."):
        if normalized.startswith(prefix):
            normalized = normalized.removeprefix(prefix)
            break
    normalized = re.sub(r"_\d+$", "", normalized)
    if normalized.endswith("_") and not normalized.endswith("__"):
        normalized = normalized[:-1]
    return normalized


def _normalize_structural_operation(operation: str) -> str | None:
    if "scaled_dot_product_attention" in operation or operation.endswith(".sdpa"):
        return "scaled_dot_product_attention"
    if convolution := _first_contained_alias(operation, _CONVOLUTION_ALIASES):
        return convolution
    if "linear" in operation:
        return "linear"
    if "matmul" in operation or operation in {"mm", "bmm"}:
        return "bmm" if operation == "bmm" else "matmul"
    if operation in _INDEXED_WRITES:
        return operation
    return _first_contained_alias(operation, _LOSS_ALIASES)


def _normalize_suffix_operation(operation: str) -> str | None:
    for canonical in ("add", "sub", "mul", "div"):
        if operation == canonical or operation.endswith(
            (f".{canonical}", f"_{canonical}")
        ):
            return canonical
    aliases = {
        "bitwise_and": {"bitwise_and", "__and__"},
        "masked_fill": {"masked_fill"},
        "bitwise_not": {"bitwise_not", "__invert__"},
    }
    for canonical, exact in aliases.items():
        if _matches_qualified_alias(operation, exact):
            return canonical
    for canonical, special in (
        ("eq", "__eq__"),
        ("ne", "__ne__"),
        ("lt", "__lt__"),
        ("le", "__le__"),
        ("gt", "__gt__"),
        ("ge", "__ge__"),
    ):
        if operation in {canonical, special} or operation.endswith(
            (f".{canonical}", f"_{canonical}")
        ):
            return canonical
    return None


def _normalize_reduction_operation(operation: str) -> str | None:
    if operation in _MATH_OPERATIONS or operation == "einsum":
        return operation
    for cumulative in ("cumsum", "cumprod", "cummax", "cummin"):
        if cumulative in operation:
            return cumulative
    if "sum" in operation and "logsumexp" not in operation:
        return "sum"
    if "mean" in operation:
        return "mean"
    if "prod" in operation:
        return "prod"
    if _matches_qualified_alias(operation, {"max", "amax"}):
        return "max"
    if _matches_qualified_alias(operation, {"min", "amin"}):
        return "min"
    return None


_OPERATION_NORMALIZERS: tuple[Callable[[str], str | None], ...] = (
    _normalize_structural_operation,
    _normalize_suffix_operation,
    lambda operation: _first_contained_alias(operation, _ACTIVATION_ALIASES),
    _normalize_reduction_operation,
)


def normalize_operation_name(operation: str) -> str:
    """Normalize framework and generated operation names to registry keys."""
    normalized = _strip_operation_syntax(operation)
    for normalizer in _OPERATION_NORMALIZERS:
        if canonical := normalizer(normalized):
            return canonical
    return normalized.rsplit(".", maxsplit=1)[-1]


class EinsumAnalyzer:
    """Analyzes operations and converts them to einsum notation.

    This class provides methods to convert various PyTorch operations
    (matmul, conv, attention, etc.) to einsum notation for analysis.

    The actual conversion logic is delegated to handlers registered
    in the global EinsumOpRegistry.
    """

    def __init__(self, debug: bool = False):
        """Initialize the EinsumAnalyzer.

        Args:
            debug: Enable debug output.
        """
        self.debug = debug
        self._registry = get_global_registry()

    def get_compute_cost(
        self, op_name: str, shapes: TensorShapes, **kwargs: Any
    ) -> int:
        """Get compute cost for an operation.

        Args:
            op_name: Name of the operation.
            shapes: Positional input/output tensor shapes.

        Returns:
            Number of operations required.
        """
        equation = kwargs.pop("equation", None)
        ts = TensorShapes(inputs=list(shapes.inputs), outputs=list(shapes.outputs))

        op_norm = self._get_operation_from_name(op_name)
        if op_norm in {"conv1d", "conv2d", "conv3d"}:
            if not ts.outputs and ts.num_inputs >= 2:
                input_shape = ts.inputs[0]
                weight_shape = ts.inputs[1]
                out_shape = self._infer_conv_output_shape(
                    op_norm, input_shape, weight_shape, **kwargs
                )
                if out_shape:
                    ts = TensorShapes(inputs=ts.inputs, outputs=[out_shape])

        if equation:
            return compute_cost_from_equation(str(equation), ts)

        einsum_op = self.get_einsum_op(op_name, ts, **kwargs)
        return einsum_op.get_compute_cost(ts)

    def get_memory_cost(self, shapes: Dict[str, TensorShape]) -> Dict[str, int]:
        """Calculate memory cost for tensors.

        Args:
            shapes: Dictionary of tensor shapes.

        Returns:
            Dictionary mapping tensor names to element counts.
        """
        memory_cost: Dict[str, int] = {}
        for name, shape in shapes.items():
            elements = 1
            for dim in shape:
                elements *= dim
            memory_cost[name] = elements
        memory_cost["total"] = sum(memory_cost.values())
        return memory_cost

    def get_einsum_op(
        self, op_name: str, shapes: TensorShapes, **kwargs: Any
    ) -> EinsumOp:
        """Get an einsum operation for the given operation name.

        Args:
            op_name: Name of the operation.
            shapes: Positional input/output tensor shapes.

        Returns:
            EinsumOp object.

        Raises:
            ValueError: If operation is not supported.
        """
        ts = shapes

        op_norm = self._get_operation_from_name(op_name)

        # Try to get handler from registry
        if self._registry.has_handler(op_norm):
            return self._registry.get_einsum_op(op_norm, ts, **kwargs)

        raise ValueError(f"Unsupported operation: {op_name}")

    def _get_operation_from_name(self, op_name: str) -> str:
        """Normalize an operation name to a canonical operation key."""
        return normalize_operation_name(op_name)

    def get_reduction_einsum_op(
        self,
        op_name: str,
        shapes: TensorShapes,
        reduce_dims: Optional[List[int]] = None,
        keepdim: bool = False,
    ) -> EinsumOp:
        """Get an einsum op for a reduction (sum/mean/prod)."""
        op_norm = self._get_operation_from_name(op_name)
        return self.get_einsum_op(op_norm, shapes, dims=reduce_dims, keepdim=keepdim)

    def _infer_conv_output_shape(
        self,
        op_norm: str,
        input_shape: TensorShape,
        weight_shape: TensorShape,
        **kwargs: Any,
    ) -> Optional[TensorShape]:
        """Infer output shape for conv ops when not provided."""
        try:
            if op_norm == "conv1d":
                b, _c, l = input_shape
                o, _c2, k = weight_shape
                stride_1d = int((kwargs.get("stride") or (1,))[0])
                padding_1d = int((kwargs.get("padding") or (0,))[0])
                dilation_1d = int((kwargs.get("dilation") or (1,))[0])
                l_out = (
                    l + 2 * padding_1d - dilation_1d * (k - 1) - 1
                ) // stride_1d + 1
                return [b, o, l_out]

            if op_norm == "conv3d":
                b, _c, d, h, w = input_shape
                o, _c2, kd, kh, kw = weight_shape
                stride = tuple(kwargs.get("stride") or (1, 1, 1))
                padding = tuple(kwargs.get("padding") or (0, 0, 0))
                dilation = tuple(kwargs.get("dilation") or (1, 1, 1))
                d_out = (d + 2 * padding[0] - dilation[0] * (kd - 1) - 1) // stride[
                    0
                ] + 1
                h_out = (h + 2 * padding[1] - dilation[1] * (kh - 1) - 1) // stride[
                    1
                ] + 1
                w_out = (w + 2 * padding[2] - dilation[2] * (kw - 1) - 1) // stride[
                    2
                ] + 1
                return [b, o, d_out, h_out, w_out]

            # Default conv2d.
            b, _c, h, w = input_shape
            o, _c2, kh, kw = weight_shape
            stride = tuple(kwargs.get("stride") or (1, 1))
            padding = tuple(kwargs.get("padding") or (0, 0))
            dilation = tuple(kwargs.get("dilation") or (1, 1))
            h_out = (h + 2 * padding[0] - dilation[0] * (kh - 1) - 1) // stride[0] + 1
            w_out = (w + 2 * padding[1] - dilation[1] * (kw - 1) - 1) // stride[1] + 1
            return [b, o, h_out, w_out]
        except Exception:
            return None

    def get_torch_einsum_equation(
        self, op_name: str, shapes: Optional[TensorShapes] = None
    ) -> str:
        """Get torch einsum equation string for an operation.

        Args:
            op_name: Name of the operation.
            shapes: Optional dictionary of tensor shapes.

        Returns:
            Einsum equation string.
        """
        if not shapes:
            # Return generic equation based on operation type
            op_lower = op_name.lower()
            if "matmul" in op_lower:
                return "ij,jk->ik"
            elif "linear" in op_lower:
                return "...k,nk->...n"
            elif "conv2d" in op_lower:
                return "bchw,ocrs->bopq"  # R,S are kernel dims, P,Q are output spatial dims
            elif "conv3d" in op_lower:
                return "bcdhw,octrs->bopqu"  # T,R,S are kernel dims, P,Q,U are output spatial dims
            else:
                return ""

        einsum_op = self.get_einsum_op(op_name, shapes)
        return einsum_op.equation

    # =========================================================================
    # Backward compatibility methods
    # =========================================================================

    def generate_matmul_einsum(
        self, input_shape: TensorShape, other_shape: TensorShape
    ) -> EinsumOp:
        """Generate einsum for matrix multiplication (backward compatibility)."""
        return self.get_einsum_op(
            "matmul", TensorShapes(inputs=[input_shape, other_shape])
        )

    def generate_linear_einsum(
        self, input_shape: TensorShape, weight_shape: TensorShape
    ) -> EinsumOp:
        """Generate einsum for linear layer (backward compatibility)."""
        return self.get_einsum_op(
            "linear", TensorShapes(inputs=[input_shape, weight_shape])
        )

    def generate_conv2d_einsum(
        self,
        input_shape: TensorShape,
        weight_shape: TensorShape,
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0),
        dilation: Tuple[int, int] = (1, 1),
    ) -> EinsumOp:
        """Generate einsum for 2D convolution (backward compatibility)."""
        return self.get_einsum_op(
            "conv2d",
            TensorShapes(inputs=[input_shape, weight_shape]),
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

    def generate_conv1d_einsum(
        self,
        input_shape: TensorShape,
        weight_shape: TensorShape,
        stride: Tuple[int] = (1,),
        padding: Tuple[int] = (0,),
        dilation: Tuple[int] = (1,),
    ) -> EinsumOp:
        """Generate einsum for 1D convolution (backward compatibility)."""
        return self.get_einsum_op(
            "conv1d",
            TensorShapes(inputs=[input_shape, weight_shape]),
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

    def generate_conv3d_einsum(
        self,
        input_shape: TensorShape,
        weight_shape: TensorShape,
        stride: Tuple[int, int, int] = (1, 1, 1),
        padding: Tuple[int, int, int] = (0, 0, 0),
        dilation: Tuple[int, int, int] = (1, 1, 1),
    ) -> EinsumOp:
        """Generate einsum for 3D convolution (backward compatibility)."""
        return self.get_einsum_op(
            "conv3d",
            TensorShapes(inputs=[input_shape, weight_shape]),
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

    def generate_elementwise_einsum(
        self, shape: TensorShape, op_type: str = "elementwise"
    ) -> EinsumOp:
        """Generate einsum for elementwise operations (backward compatibility)."""
        return self.get_einsum_op(op_type, TensorShapes(inputs=[shape]))

    def generate_binary_elementwise_einsum(
        self, input_shape: TensorShape, input_1_shape: TensorShape, op_type: str = "add"
    ) -> EinsumOp:
        """Generate einsum for binary elementwise operations (backward compatibility)."""
        return self.get_einsum_op(
            op_type, TensorShapes(inputs=[input_shape, input_1_shape])
        )

    def generate_reduction_einsum(
        self,
        shape: TensorShape,
        op_type: str = "sum",
        dims: Optional[List[int]] = None,
        keepdim: bool = False,
    ) -> EinsumOp:
        """Generate einsum for reduction operations (backward compatibility)."""
        return self.get_einsum_op(
            op_type, TensorShapes(inputs=[shape]), dims=dims, keepdim=keepdim
        )


__all__ = ["EinsumAnalyzer"]

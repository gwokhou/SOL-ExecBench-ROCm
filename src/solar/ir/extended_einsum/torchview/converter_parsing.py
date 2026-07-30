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

"""Convert PyTorch computation graphs to einsum representation.

This module implements the first stage of the Solar pipeline:

    pytorch_graph.yaml -> einsum_graph.yaml -> einsum_graph_renamed.yaml

The output follows the einsum graph schema:

    layers:
      <layer_id>:
        type: <operation_type>
        einsum_equation: <equation_string>
        elementwise_op: <op>
        reduction_op: <op>
        is_real_einsum: <bool>
        is_einsum_supportable: <bool>
        tensor_names: {inputs: [...], outputs: [...]}
        tensor_shapes: {inputs: [...], outputs: [...]}
        connections: {inputs: [...], outputs: [...]}

Example:
    >>> from solar.ir.extended_einsum.torchview.converter import PyTorchToEinsum
    >>> converter = PyTorchToEinsum()
    >>> result = converter.convert("input/pytorch_graph.yaml", "output/")
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any

PathLike = str | Path

from solar.ir.extended_einsum.torchview.converter_contract import (
    ConverterMixinContract,
)
from solar.ir.extended_einsum.torchview.quirk_repair import (
    TorchviewRepairContext,
    repair_torchview_quirks,
)


class ConverterParsingMixin(ConverterMixinContract):
    """Parse raw Torchview attributes and repair metadata quirks."""

    def _parse_einsum_from_raw_attributes(
        self,
        module_args: dict[str, Any],
    ) -> str | None:
        r"""Parse einsum equation from raw_attributes in module_args.

        For torch.einsum operations, the raw_attributes field contains the
        einsum equation string as the first argument.

        Example raw_attributes:
            "[[\'bijl,lk->bijk\', Tensor(...), Tensor(...)], {}]"

        Args:
            module_args: Dictionary containing module arguments.

        Returns:
            Solar-compatible einsum equation (uppercase) or None if not found.
        """
        raw_attrs = module_args.get("raw_attributes", "")
        if not raw_attrs:
            return None

        # Pattern: first string argument in the list, e.g., 'bijl,lk->bijk'
        import re

        # Match quoted string that looks like an einsum equation (contains -> and comma)
        pattern = r"['\"]([a-zA-Z0-9,\s]+->[\s]*[a-zA-Z0-9]+)['\"]"
        match = re.search(pattern, raw_attrs)

        if match:
            equation = match.group(1).strip()
            return self._convert_einsum_to_solar_format(equation)

        return None

    def _convert_einsum_to_solar_format(self, equation: str) -> str:
        """Convert a lowercase einsum equation to Solar's uppercase format.

        Solar uses uppercase letters for dimension labels, with optional
        numeric suffixes for batch dimensions (e.g., B0, B1).

        Example:
            'bijl,lk->bijk' -> 'B0IJL,LK->B0IJK'

        Args:
            equation: Lowercase einsum equation string.

        Returns:
            Uppercase einsum equation string.
        """
        if not equation or "->" not in equation:
            return equation

        # Split into inputs and output
        parts = equation.split("->")
        if len(parts) != 2:
            return equation.upper()

        lhs, rhs = parts[0].strip(), parts[1].strip()

        all_dims = set()
        for char in lhs + rhs:
            if char.isalpha():
                all_dims.add(char.lower())

        # Map lowercase dimensions to Solar's uppercase convention.
        dim_map = {d: d.upper() for d in sorted(all_dims)}

        # Apply mapping to equation
        result_lhs = ""
        for char in lhs:
            if char.isalpha():
                result_lhs += dim_map.get(char.lower(), char.upper())
            else:
                result_lhs += char

        result_rhs = ""
        for char in rhs:
            if char.isalpha():
                result_rhs += dim_map.get(char.lower(), char.upper())
            else:
                result_rhs += char

        return f"{result_lhs}->{result_rhs}"

    def _parse_reduction_args_from_raw_attributes(
        self,
        module_args: dict[str, Any],
    ) -> tuple[list[int] | None, bool]:
        """Parse reduction arguments (dim, keepdim) from raw_attributes.

        For reduction operations like sum/mean/max/min, the raw_attributes field
        contains the dim and keepdim arguments.

        Example raw_attributes:
            "[[Tensor(...)], {dim: 1}]"
            "[[Tensor(...)], {dim: 1, keepdim: True}]"
            "[[Tensor(...)], {dim: [1, 2]}]"

        Args:
            module_args: Dictionary containing module arguments.

        Returns:
            Tuple of (reduction_dims, keepdim). reduction_dims is a list of ints or None.
        """
        reduce_dims: list[int] | None
        # First check parsed dim/keepdim fields (from _parse_torchview_attributes)
        if "dim" in module_args:
            dim_val = module_args["dim"]
            reduce_dims = (
                [dim_val] if isinstance(dim_val, int) else list(dim_val)
            )
            keepdim = bool(module_args.get("keepdim", False))
            return reduce_dims, keepdim

        # Then try raw_attributes string (regex parsing)
        raw_attrs = module_args.get("raw_attributes", "")
        if not raw_attrs:
            return None, False

        reduce_dims = None
        keepdim = False

        # Match dim: [<numbers>] first (list case)
        list_dim_pattern = r"dim:\s*\[([^\]]+)\]"
        match = re.search(list_dim_pattern, raw_attrs)
        if match:
            dims_str = match.group(1)
            reduce_dims = [int(d.strip()) for d in dims_str.split(",")]
        else:
            # Pattern for single dim: dim: 1 or dim: -1
            single_dim_pattern = r"dim:\s*(-?\d+)"
            match = re.search(single_dim_pattern, raw_attrs)
            if match:
                reduce_dims = [int(match.group(1))]

        # Match keepdim: True or keepdim: False
        keepdim_pattern = r"keepdim:\s*(True|False)"
        match = re.search(keepdim_pattern, raw_attrs)
        if match:
            keepdim = match.group(1) == "True"

        return reduce_dims, keepdim

    def _tensor_arg_shapes_from_raw(
        self,
        module_args: dict[str, Any],
    ) -> list[tuple[int, ...] | None]:
        """Shapes of the positional ``Tensor`` arguments recorded by torchview.

        torchview stores the real call signature in ``raw_attributes``, e.g.::

            [
                [
                    Tensor(shape=(112, 64, 512, 512), dtype=torch.float32),
                    Tensor(shape=(), dtype=torch.float32),
                ],
                {p: "fro"},
            ]

        Returns the shape tuple of every ``Tensor(shape=(...))`` occurrence in
        order; a scalar tensor yields ``()`` and an unparseable shape yields
        ``None``. This is ground truth for an op's true tensor arity — used by
        the dropped-edge repair in ``_build_op_graph`` to detect ops whose
        recorded ``input_shapes`` undercount their actual tensor inputs
        (torchview can drop a scalar-tensor edge, e.g. ``x / x.norm()``).
        """
        raw = module_args.get("raw_attributes", "") if module_args else ""
        if not raw:
            return []
        shapes: list[tuple[int, ...] | None] = []
        for m in re.finditer(r"Tensor\(shape=\(([^)]*)\)", raw):
            body = m.group(1).strip().rstrip(",").strip()
            if not body:
                shapes.append(())
                continue
            try:
                shapes.append(
                    tuple(int(d.strip()) for d in body.split(",") if d.strip())
                )
            except ValueError:
                shapes.append(None)
        return shapes

    @staticmethod
    def _tensor_arg_dtypes_from_raw(module_args: dict[str, Any]) -> list[str]:
        """Return exact positional tensor dtypes recorded by torchview."""
        raw = module_args.get("raw_attributes", "") if module_args else ""
        return [
            f"torch.{match}"
            for match in re.findall(
                r"Tensor\(shape=\([^)]*\),\s*dtype=torch\.([A-Za-z0-9_]+)\)",
                raw,
            )
        ]

    @classmethod
    def _bits_of_dtype(cls, dtype_str: str | None) -> int:
        if not dtype_str:
            return 32
        return cls._DTYPE_BITS.get(
            str(dtype_str).replace("torch.", "").lower(), 32
        )

    def _repair_torchview_quirks(
        self,
        layers: dict[str, Any],
        op_ids: list[str],
        tensor_ids: list[str],
    ) -> None:
        """Repair known Torchview quirks through ordered typed passes."""
        context = TorchviewRepairContext(
            layers=layers,
            operation_ids=tuple(op_ids),
            tensor_ids=tuple(tensor_ids),
            parameter_tensor_indices=self._PARAMETER_TENSOR_INDICES,
            output_dtype_input_index=self._OUTPUT_DTYPE_INPUT_INDEX,
            shape_op_types_for_dtype=self._SHAPE_OP_TYPES_FOR_DTYPE,
            parse_shapes=self._tensor_arg_shapes_from_raw,
            parse_dtypes=self._tensor_arg_dtypes_from_raw,
            dtype_bits=self._bits_of_dtype,
        )
        repair_torchview_quirks(context)
        self._tensor_to_producer_op = context.tensor_to_producer
        self._tensor_to_producer_slot = context.tensor_to_producer_slot

    def _validate_tensor_shape_consistency(
        self,
        einsum_graph: dict[str, Any],
    ) -> None:
        """Assert that every tensor name reused across einsums has a single shape.

        Walks every einsum's tensor_names + tensor_shapes (inputs and outputs)
        and builds a (name → set of shape tuples) map. Any name with >1 distinct
        shape indicates an emit bug — for example, the cumsum_exclusive case
        where ``Model.cat`` mis-attributed input 0 to ``Model.cumsum``, causing
        the name ``Model.cumsum.Output`` to claim both ``[32768, 1]`` and
        ``[32768, 32767]`` shapes.

        Surfaces the violation at the solar boundary (with a list of every
        einsum that referenced the conflicting tensor) rather than letting
        the inconsistency propagate to AccelForge.
        """
        shapes_by_name: dict[str, dict[tuple[int, ...], list[str]]] = (
            defaultdict(lambda: defaultdict(list))
        )
        for layer_name, m in (einsum_graph.get("layers") or {}).items():
            tnames = m.get("tensor_names") or {}
            tshapes = m.get("tensor_shapes") or {}
            for side in ("inputs", "outputs"):
                names = tnames.get(side) or []
                shapes = tshapes.get(side) or []
                n = min(len(names), len(shapes))
                for i in range(n):
                    if shapes[i] is None:
                        continue
                    key = tuple(shapes[i])
                    shapes_by_name[names[i]][key].append(
                        f"{layer_name}.{side}[{i}]"
                    )
        conflicts: list[str] = []
        for name, by_shape in shapes_by_name.items():
            if len(by_shape) > 1:
                listed = "; ".join(
                    f"{list(shape)} via {refs}"
                    for shape, refs in by_shape.items()
                )
                conflicts.append(f"  '{name}': {listed}")
        if conflicts:
            raise ValueError(
                "Pre-AF tensor-shape inconsistency detected: a tensor name "
                "is referenced with multiple distinct shapes across einsums. "
                "This usually means a producer was mis-attributed in stage-1. "
                "Conflicts:\n" + "\n".join(conflicts)
            )

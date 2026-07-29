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
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PathLike = str | Path

from solar.ir.extended_einsum.torchview.converter_contract import (
    ConverterMixinContract,
)
from solar.ir.extended_einsum.torchview.converter_models import ConversionError


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
        """Single pass repairing every known torchview tracing quirk.

        torchview is great at recording shapes and types but consistently
        drops a small number of well-known patterns. Three repairs run
        here in dependency order so every downstream pass — per-op
        handlers, op-graph build, AF emission — sees a clean ``layers``
        dict and never has to second-guess torchview:

        **(A) Dropped scalar-tensor edges** (FrobeniusNorm pattern).
        torchview can omit a tensor edge entirely when one operand is a
        scalar tensor produced by a reduction (canonical: ``x / x.norm()``).
        Detected by comparing each op's ``raw_attributes`` tensor-arg count
        against its recorded ``input_shapes``; missing slots are wired in
        from a matching dangling ``hidden-tensor`` dead-end (one producer,
        no consumer) of the same shape.

        **(B) Split tensor-node pairs** (cumsum_exclusive zeros_like).
        torchview occasionally splits a single physical tensor flowing
        producer→consumer into two disconnected nodes — an orphan (no
        producer, has consumer) and a dead-end (has producer, no consumer)
        of matching ``(shape, dtype)``. We rewire the orphan's
        ``connections.inputs`` to the dead-end's producer so the normal
        edge build picks up the producer→consumer edge, and record the
        producer in ``self._tensor_to_producer_op`` for the converter's
        hidden-tensor resolution path.

        **(C) Wrong output dtypes** (fp32 override on bf16/fp16 outputs).
        torchview reports ``torch.float32`` on FunctionNode outputs even
        when all inputs are narrower. Walk the layers in insertion (topo)
        order; for each non-shape op set the output dtype to the widest
        input dtype and propagate forward through downstream tensor nodes
        and the consumer ops' ``input_dtypes``. Shape ops pass through the
        first input's dtype.

        Mutates ``layers`` in place. Initializes
        ``self._tensor_to_producer_op``. Conservative: only unambiguous
        1:1 matches are applied for (A) and (B); ambiguous candidates are
        left untouched.
        """
        self._tensor_to_producer_op = {}
        self._tensor_to_producer_slot = {}
        op_id_set = set(op_ids)

        # torchview may omit module parameters from the bipartite graph even
        # though its exact call record still contains them. Materialize those
        # tensor arguments as explicit producerless weight inputs. They remain
        # external graph inputs and therefore must be supplied by any exact
        # replay; parameter values are never embedded or guessed.
        for op_id in op_ids:
            odata = layers.get(op_id) or {}
            op_type = str(odata.get("type", "")).lower()
            parameter_indices = self._PARAMETER_TENSOR_INDICES.get(
                op_type, set()
            )
            if not parameter_indices:
                continue
            raw_shapes = self._tensor_arg_shapes_from_raw(
                odata.get("module_args") or {}
            )
            raw_dtypes = self._tensor_arg_dtypes_from_raw(
                odata.get("module_args") or {}
            )
            input_shapes = odata.setdefault("input_shapes", [])
            input_dtypes = odata.setdefault("input_dtypes", [])
            input_types = odata.setdefault("input_types", [])
            input_connections = odata.setdefault("connections", {}).setdefault(
                "inputs", []
            )
            for index in sorted(parameter_indices):
                if index < len(input_shapes) or index >= len(raw_shapes):
                    continue
                shape = raw_shapes[index]
                if shape is None or index >= len(raw_dtypes):
                    raise ConversionError(
                        f"cannot recover exact parameter metadata for {op_id}"
                    )
                if index != len(input_shapes):
                    raise ConversionError(
                        f"parameter tensor order is incomplete for {op_id}"
                    )
                input_shapes.append(list(shape))
                input_dtypes.append(raw_dtypes[index])
                input_types.append("weight")
                input_connections.append(f"{op_id}.auxiliary-tensor_{index}")

        # Torchview tensor nodes can carry the process default dtype instead
        # of the actual dtype (notably integer gather indices).  The recorded
        # call signature contains the runtime dtype of every tensor argument,
        # so repair both the consumer slots and their source tensor nodes.
        for op_id in op_ids:
            odata = layers.get(op_id) or {}
            raw_dtypes = self._tensor_arg_dtypes_from_raw(
                odata.get("module_args") or {}
            )
            input_shapes = odata.get("input_shapes") or []
            if raw_dtypes and len(raw_dtypes) == len(input_shapes):
                odata["input_dtypes"] = list(raw_dtypes)
                input_tensors = (odata.get("connections") or {}).get(
                    "inputs"
                ) or []
                if len(input_tensors) == len(raw_dtypes):
                    for tensor_id, dtype in zip(
                        input_tensors, raw_dtypes, strict=False
                    ):
                        if tensor_id in layers and tensor_id not in op_id_set:
                            tensor = layers[tensor_id]
                            output_count = (
                                len(tensor.get("output_shapes") or []) or 1
                            )
                            tensor["output_dtypes"] = [dtype] * output_count

        # --- (A+B prep) Index orphan / dead-end tensor nodes --------------
        # The shape/dtype of a tensor node is normally its own
        # ``output_shapes[0]`` / ``output_dtypes[0]``. But torchview's
        # ``hidden-tensor`` placeholder records empty lists for both, so we
        # fall back to the producer's ``output_shapes[0]`` when the tensor
        # itself has no shape recorded.
        orphans_by_key: dict[tuple[tuple[int, ...], str], list[str]] = (
            defaultdict(list)
        )
        hidden_dangling_by_shape: dict[
            tuple[int, ...], list[tuple[str, str, str]]
        ] = defaultdict(list)
        for tensor_id in tensor_ids:
            tdata = layers.get(tensor_id) or {}
            conns = tdata.get("connections") or {}
            producers_ = [
                p for p in (conns.get("inputs") or []) if p in op_id_set
            ]
            consumers_ = [
                c for c in (conns.get("outputs") or []) if c in op_id_set
            ]
            shapes = (
                tdata.get("output_shapes") or tdata.get("input_shapes") or []
            )
            dtypes = (
                tdata.get("output_dtypes") or tdata.get("input_dtypes") or []
            )
            if not shapes and len(producers_) == 1:
                pdata = layers.get(producers_[0]) or {}
                shapes = pdata.get("output_shapes") or []
                dtypes = pdata.get("output_dtypes") or []
            if not shapes:
                continue
            sh = tuple(shapes[0]) if shapes[0] is not None else ()
            dt = str(dtypes[0]) if dtypes else ""
            key = (sh, dt)
            if not producers_ and consumers_:
                orphans_by_key[key].append(tensor_id)
            elif (
                len(producers_) == 1
                and not consumers_
                and (tdata.get("type") or "").lower() == "hidden-tensor"
            ):
                hidden_dangling_by_shape[sh].append(
                    (tensor_id, producers_[0], dt)
                )

        # --- (A) Dropped scalar-tensor edges ------------------------------
        consumed: set[tuple[str, str]] = set()
        for op_id in op_ids:
            odata = layers.get(op_id) or {}
            arg_shapes = [
                s
                for s in self._tensor_arg_shapes_from_raw(
                    odata.get("module_args") or {}
                )
                if s is not None
            ]
            if not arg_shapes:
                continue
            recorded = [
                tuple(s)
                for s in (odata.get("input_shapes") or [])
                if s is not None
            ]
            missing = Counter(arg_shapes) - Counter(recorded)
            if not missing:
                continue
            in_dtypes = odata.get("input_dtypes") or []
            default_dt = str(in_dtypes[0]) if in_dtypes else "torch.float32"
            for sh, cnt in missing.items():
                for _ in range(cnt):
                    candidates = [
                        (t, p, d)
                        for (t, p, d) in hidden_dangling_by_shape.get(sh, [])
                        if (t, p) not in consumed
                    ]
                    if len(candidates) != 1:
                        continue  # ambiguous or none
                    tensor_id, producer_op, dt = candidates[0]
                    if producer_op == op_id:
                        continue
                    consumed.add((tensor_id, producer_op))
                    odata.setdefault("input_shapes", []).append(list(sh))
                    odata.setdefault("input_dtypes", []).append(
                        dt or default_dt
                    )
                    odata.setdefault("input_types", []).append("input")
                    oc = odata.setdefault("connections", {}).setdefault(
                        "inputs", []
                    )
                    if tensor_id not in oc:
                        oc.append(tensor_id)
                    tdata = layers.get(tensor_id) or {}
                    tout = tdata.setdefault("connections", {}).setdefault(
                        "outputs", []
                    )
                    if op_id not in tout:
                        tout.append(op_id)

        # Tensor-valued keyword arguments (for example an SDPA mask) can be
        # present in the exact signature without a torchview edge. After the
        # producer-repair pass above has claimed every unambiguous internal
        # tensor, preserve remaining suffix arguments as explicit external
        # inputs. This is fail-closed: replay must supply their values.
        for op_id in op_ids:
            odata = layers.get(op_id) or {}
            raw_shapes = self._tensor_arg_shapes_from_raw(
                odata.get("module_args") or {}
            )
            raw_dtypes = self._tensor_arg_dtypes_from_raw(
                odata.get("module_args") or {}
            )
            input_shapes = odata.setdefault("input_shapes", [])
            input_dtypes = odata.setdefault("input_dtypes", [])
            input_types = odata.setdefault("input_types", [])
            input_connections = odata.setdefault("connections", {}).setdefault(
                "inputs", []
            )
            if len(input_shapes) > len(raw_shapes):
                continue
            parameter_indices = self._PARAMETER_TENSOR_INDICES.get(
                str(odata.get("type", "")).lower(), set()
            )
            for index in range(len(input_shapes), len(raw_shapes)):
                shape = raw_shapes[index]
                if shape is None or index >= len(raw_dtypes):
                    raise ConversionError(
                        f"cannot recover exact tensor argument metadata for {op_id}"
                    )
                input_shapes.append(list(shape))
                input_dtypes.append(raw_dtypes[index])
                input_types.append(
                    "weight" if index in parameter_indices else "input"
                )
                synthetic_id = f"{op_id}.auxiliary-tensor_{index}"
                input_connections.append(synthetic_id)
                if (
                    index not in parameter_indices
                    and synthetic_id not in layers
                ):
                    layers[synthetic_id] = {
                        "type": "auxiliary-tensor",
                        "node_class": "TensorNode",
                        "input_shapes": [],
                        "output_shapes": [list(shape)],
                        "input_dtypes": [],
                        "output_dtypes": [raw_dtypes[index]],
                        "input_types": [],
                        "output_types": ["output"],
                        "module_args": {
                            "hierarchical_name": synthetic_id,
                            "recovered_from": "exact_call_signature",
                        },
                        "connections": {"inputs": [], "outputs": [op_id]},
                    }
            if raw_dtypes and len(raw_dtypes) == len(input_shapes):
                odata["input_dtypes"] = list(raw_dtypes)

        # --- (B) Split tensor-node pairs ----------------------------------
        for key, orphan_ids in orphans_by_key.items():
            de_list = self._matching_hidden_dangling(
                key,
                hidden_dangling_by_shape,
                consumed,
            )
            if len(orphan_ids) != 1 or len(de_list) != 1:
                continue
            orphan_id = orphan_ids[0]
            producer_op = de_list[0][1]
            self._tensor_to_producer_op[orphan_id] = producer_op
            # Rewire the orphan's ``connections.inputs`` to the producer so
            # the normal edge build picks up producer→consumer naturally.
            orphan_data = layers.get(orphan_id) or {}
            ocon_in = orphan_data.setdefault("connections", {}).setdefault(
                "inputs", []
            )
            if producer_op not in ocon_in:
                ocon_in.append(producer_op)

        # --- (C) Output-dtype correction ----------------------------------
        # Seed from every non-operation node; operation outputs are corrected
        # below in topological order.
        corrected_dtype: dict[str, str] = {}
        for layer_id, ldata in layers.items():
            if layer_id in op_id_set:
                continue
            outd = ldata.get("output_dtypes") or ldata.get("input_dtypes") or []
            if outd:
                corrected_dtype[layer_id] = outd[0]

        for layer_id, odata in layers.items():
            if layer_id not in op_id_set:
                continue
            in_tensors = (odata.get("connections") or {}).get("inputs") or []
            in_dtypes = list(odata.get("input_dtypes") or [])
            for i, tid in enumerate(in_tensors):
                if tid in corrected_dtype and i < len(in_dtypes):
                    in_dtypes[i] = corrected_dtype[tid]
            if in_dtypes:
                odata["input_dtypes"] = in_dtypes
            layer_type = (odata.get("type") or "").lower()
            if self._correct_topk_output_dtypes(
                layers,
                odata,
                corrected_dtype,
            ):
                continue
            dtype_methods = {
                "bfloat16": "torch.bfloat16",
                "float": "torch.float32",
                "half": "torch.float16",
                "int": "torch.int32",
                "long": "torch.int64",
            }
            if layer_type in dtype_methods:
                widest = dtype_methods[layer_type]
            elif layer_type in {
                "eq",
                "ne",
                "lt",
                "le",
                "gt",
                "ge",
                "__eq__",
                "__ne__",
                "__lt__",
                "__le__",
                "__gt__",
                "__ge__",
            }:
                widest = "torch.bool"
            elif layer_type in {"bitwise_and", "__and__"}:
                widest = (
                    in_dtypes[0]
                    if in_dtypes
                    else (odata.get("output_dtypes") or ["torch.bool"])[0]
                )
            elif layer_type == "to":
                requested_dtype = next(
                    (
                        argument["dtype"]
                        for argument in (
                            (odata.get("module_args") or {}).get(
                                "call_arguments"
                            )
                            or []
                        )
                        if isinstance(argument, dict) and "dtype" in argument
                    ),
                    None,
                )
                if requested_dtype is None:
                    widest = (
                        in_dtypes[0]
                        if in_dtypes
                        else (odata.get("output_dtypes") or ["torch.float32"])[
                            0
                        ]
                    )
                else:
                    widest = (
                        f"torch.{str(requested_dtype).removeprefix('torch.')}"
                    )
            elif (
                layer_type in self._OUTPUT_DTYPE_INPUT_INDEX
                and len(in_dtypes) > self._OUTPUT_DTYPE_INPUT_INDEX[layer_type]
            ):
                widest = in_dtypes[self._OUTPUT_DTYPE_INPUT_INDEX[layer_type]]
            elif layer_type in self._SHAPE_OP_TYPES_FOR_DTYPE:
                widest = (
                    in_dtypes[0]
                    if in_dtypes
                    else (odata.get("output_dtypes") or ["torch.float32"])[0]
                )
            elif in_dtypes:
                widest = max(in_dtypes, key=self._bits_of_dtype)
            else:
                widest = (odata.get("output_dtypes") or ["torch.float32"])[0]
            widest = str(widest or "torch.float32")
            n_out = len(odata.get("output_dtypes") or []) or 1
            odata["output_dtypes"] = [widest] * n_out
            for tid in (odata.get("connections") or {}).get("outputs") or []:
                if tid in layers:
                    tdata = layers[tid]
                    n = len(tdata.get("output_dtypes") or []) or 1
                    tdata["output_dtypes"] = [widest] * n
                corrected_dtype[tid] = widest

    @staticmethod
    def _matching_hidden_dangling(
        key: tuple[tuple[int, ...], str],
        candidates: dict[tuple[int, ...], list[tuple[str, str, str]]],
        consumed: set[tuple[str, str]],
    ) -> list[tuple[str, str]]:
        """Return hidden producer tensors matching one orphan exactly."""
        shape, dtype = key
        return [
            (tensor_id, producer)
            for tensor_id, producer, candidate_dtype in candidates.get(
                shape,
                [],
            )
            if candidate_dtype == dtype
            if (tensor_id, producer) not in consumed
        ]

    @staticmethod
    def _correct_topk_output_dtypes(
        layers: dict[str, Any],
        operation: dict[str, Any],
        corrected_dtype: dict[str, str],
    ) -> bool:
        """Preserve distinct values/index dtypes for a top-k operation."""
        if (operation.get("type") or "").lower() != "topk":
            return False
        slot_dtypes = list(operation.get("output_dtypes") or [])
        outputs = list(
            (operation.get("connections") or {}).get("outputs") or [],
        )
        if len(slot_dtypes) != 2 or len(outputs) != 2:
            raise ValueError("topk requires two exact output dtype slots")
        for index, tensor_id in enumerate(outputs):
            if tensor_id in layers:
                tensor_data = layers[tensor_id]
                count = len(tensor_data.get("output_dtypes") or []) or 1
                tensor_data["output_dtypes"] = [slot_dtypes[index]] * count
            corrected_dtype[tensor_id] = slot_dtypes[index]
        return True

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

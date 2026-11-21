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

"""Analyze a SOLAR IR graph into hardware-independent compute and I/O metrics.

This is the second SOLAR pipeline stage, converting an IR graph artifact into
``analysis.yaml``. It emits per-layer and graph totals plus conservative formal
fusion and Orojenesis evidence. See ``SOL_GUIDE.md`` for the memory models and
formal-evidence contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from solar.analysis.graph_context import (
    GraphTopology,
    PreparedAnalysis,
)
from solar.analysis.graph_models import (
    AnalysisAccumulator,
    AnalyzedLayer,
    GraphIoTotals,
    InputIo,
    IntrinsicLayer,
    LayerCompute,
    LayerData,
    LayerIo,
    MemoryBytes,
    MemoryElements,
    OutputIo,
    ResourceAccounting,
)
from solar.analysis.output_writes import ExternalSlotWrite
from solar.analysis.resources import (
    classify_layer_resources,
    merge_resource_work,
)
from solar.composition import BoundComponent
from solar.ir.contracts import layer_operation, resolve_operand_tensor_slots
from solar.precision import (
    BYTES_PER_ELEMENT,
    normalize_dtype,
)
from solar.schema_versions import (
    ResourceModelVersion as SolarResourceModelVersion,
)


def _serialized_tensor_metadata(
    data: LayerData,
    memory: MemoryElements,
    byte_counts: MemoryBytes,
) -> dict[str, object]:
    """Return the shape, size, identity, dtype, and memory tensor payload."""
    return {
        "tensor_shapes": {
            "inputs": [
                shape for shape in data.input_shapes if isinstance(shape, list)
            ],
            "outputs": [
                shape for shape in data.output_shapes if isinstance(shape, list)
            ],
        },
        "tensor_sizes": {
            "inputs": data.input_sizes,
            "outputs": data.output_sizes,
        },
        "memory_elements": {
            "inputs": memory.reads,
            "outputs": memory.writes,
        },
        "memory_bytes": {
            "inputs": byte_counts.input_bytes,
            "outputs": byte_counts.output_bytes,
        },
        "tensor_dtypes": {
            "inputs": data.input_dtypes,
            "outputs": data.output_dtypes,
        },
        "tensor_names": {
            "inputs": data.input_names,
            "outputs": data.output_names,
        },
        "tensor_types": {
            "inputs": data.input_types,
            "outputs": data.output_types,
        },
    }


class LayerAccountant(BoundComponent):
    """Account graph compute, memory, and external I/O."""

    def _account_layer_resources(
        self,
        data: LayerData,
        compute: LayerCompute,
        topology: GraphTopology,
        prepared: PreparedAnalysis,
        accumulator: AnalysisAccumulator,
        *,
        orphaned: bool,
    ) -> ResourceAccounting:
        compute_precision, resource_precision = self._resource_precision(
            data,
            compute,
            topology,
            prepared,
        )
        if compute.macs:
            accumulator.macs_by_precision[compute_precision] += compute.macs
        resources = classify_layer_resources(
            data.layer,
            macs=compute.macs,
            fallback_precision=prepared.fallback_precision,
            strict=prepared.strict,
            compute_precision=resource_precision,
        )
        if orphaned:
            resources = {
                "model_version": SolarResourceModelVersion.AMD.value,
                "work": {},
                "classification": "exempt",
                "exemption_reason": "orphaned_dead_end",
                "formulas": [],
            }
        layer_work = resources.get("work")
        if not isinstance(layer_work, Mapping) or any(
            not isinstance(value, Mapping) for value in layer_work.values()
        ):
            raise TypeError(
                f"layer {data.layer_id!r} resource work is not a mapping",
            )
        merge_resource_work(
            accumulator.resource_work,
            cast(Mapping[str, Mapping[str, Any]], layer_work),
        )
        accumulator.resource_coverage[str(resources["classification"])] += 1
        return ResourceAccounting(
            compute_precision=compute_precision, resources=resources
        )

    @staticmethod
    def _contraction_payload_names(
        data: LayerData,
        semantic: Mapping[str, Any],
    ) -> list[str]:
        if semantic.get("kind") == "einsum":
            equation = str(semantic.get("equation") or data.equation or "")
            lhs = equation.split("->", 1)[0] if "->" in equation else ""
            arity = len(lhs.split(",")) if lhs else len(data.input_names)
            slots = resolve_operand_tensor_slots(
                semantic,
                tuple(range(arity)),
                len(data.input_names),
            )
            if slots is not None:
                return [data.input_names[slot] for slot in slots]
            return list(data.input_names)
        target = str(semantic.get("target") or data.op_type).lower()
        positions: tuple[int, ...] | None = None
        if target == "addmm":
            positions = (1, 2)
        elif target in {
            "mm",
            "bmm",
            "matmul",
            "linear",
            "conv1d",
            "conv2d",
            "conv3d",
            "conv_transpose1d",
            "conv_transpose2d",
            "conv_transpose3d",
        }:
            positions = (0, 1)
        elif target == "scaled_dot_product_attention":
            positions = (0, 1, 2)
        if positions is not None:
            slots = resolve_operand_tensor_slots(
                semantic,
                positions,
                len(data.input_names),
            )
            return (
                [data.input_names[slot] for slot in slots]
                if slots is not None
                else []
            )
        return list(data.input_names)

    @staticmethod
    def _resource_precision(
        data: LayerData,
        compute: LayerCompute,
        topology: GraphTopology,
        prepared: PreparedAnalysis,
    ) -> tuple[str, str | None]:
        compute_precisions = [
            "nvfp4" if normalized == "float4_e2m1fn_x2" else normalized
            for dtype in data.input_dtypes
            if (
                normalized := normalize_dtype(
                    dtype,
                    prepared.fallback_precision,
                )
            )
            in {
                "fp64",
                "fp32",
                "tf32",
                "bf16",
                "fp16",
                "fp8",
                "nvfp4",
                "float4_e2m1fn_x2",
                "int8",
                "int4",
            }
        ]
        compute_precision = (
            max(compute_precisions, key=lambda value: BYTES_PER_ELEMENT[value])
            if compute_precisions
            else prepared.fallback_precision
        )
        resource_precision: str | None = None
        semantic = layer_operation(data.layer)
        if compute.macs > 0:
            payload_names = LayerAccountant._contraction_payload_names(
                data,
                semantic,
            )
            payload_precisions = [
                topology.dequantized_payload_precision(
                    name,
                    prepared.profile,
                    prepared.fallback_precision,
                )
                for name in payload_names
            ]
            if (
                len(payload_precisions) >= 2
                and all(payload_precisions)
                and len(set(payload_precisions)) == 1
            ):
                compute_precision = str(payload_precisions[0])
                resource_precision = compute_precision
        return compute_precision, resource_precision

    @staticmethod
    def _classify_layer_inputs(
        data: LayerData,
        memory: MemoryElements,
        byte_counts: MemoryBytes,
        topology: GraphTopology,
        accumulator: AnalysisAccumulator,
    ) -> InputIo:
        intermediate_elems = 0
        model_elems = 0
        intermediate_bytes = 0.0
        model_bytes = 0.0
        for index, memory_read in enumerate(memory.reads):
            if memory_read <= 0:
                continue
            input_name = (
                data.input_names[index] if index < len(data.input_names) else ""
            )
            graph_internal = False
            if input_name in topology.tensor_producers:
                producer_id = topology.tensor_producers[input_name]
                source_id = topology.trace_source_through_views(producer_id)
                graph_internal = (
                    source_id in topology.all_layer_ids
                    and source_id not in topology.transparent_layer_ids
                )
            if graph_internal:
                intermediate_elems += memory_read
                intermediate_bytes += byte_counts.input_bytes[index]
                continue
            model_elems += memory_read
            model_bytes += byte_counts.input_bytes[index]
            if input_name:
                accumulator.unique_external_inputs[input_name] = max(
                    accumulator.unique_external_inputs.get(input_name, 0),
                    memory_read,
                )
                accumulator.unique_external_input_bytes[input_name] = max(
                    accumulator.unique_external_input_bytes.get(
                        input_name,
                        0.0,
                    ),
                    byte_counts.input_bytes[index],
                )
        return InputIo(
            intermediate_elems=intermediate_elems,
            model_elems=model_elems,
            intermediate_bytes=intermediate_bytes,
            model_bytes=model_bytes,
        )

    @staticmethod
    def _record_resolved_output_writes(
        data: LayerData,
        external_writes: Mapping[tuple[str, int], ExternalSlotWrite],
        accumulator: AnalysisAccumulator,
    ) -> tuple[int, float]:
        model_elems = 0
        model_bytes = 0.0
        for index in range(len(data.output_names)):
            slot_write = external_writes.get((data.layer_id, index))
            if slot_write is None:
                continue
            model_elems += slot_write.elements
            model_bytes += slot_write.bytes
            name = slot_write.tensor_name
            accumulator.unique_external_outputs[name] = max(
                accumulator.unique_external_outputs.get(name, 0),
                int(slot_write.elements),
            )
            accumulator.unique_external_output_bytes[name] = max(
                accumulator.unique_external_output_bytes.get(name, 0.0),
                float(slot_write.bytes),
            )
        return model_elems, model_bytes

    @staticmethod
    def _record_direct_output_writes(
        data: LayerData,
        memory: MemoryElements,
        byte_counts: MemoryBytes,
        external_flags: list[bool],
        accumulator: AnalysisAccumulator,
    ) -> tuple[int, float]:
        model_elems = sum(
            value
            for value, external in zip(
                memory.writes, external_flags, strict=True
            )
            if external
        )
        model_bytes = sum(
            value
            for value, external in zip(
                byte_counts.output_bytes, external_flags, strict=True
            )
            if external
        )
        for index, (name, external) in enumerate(
            zip(data.output_names, external_flags, strict=True),
        ):
            if not external:
                continue
            elements = memory.writes[index] if index < len(memory.writes) else 0
            bytes_ = (
                byte_counts.output_bytes[index]
                if index < len(byte_counts.output_bytes)
                else 0.0
            )
            accumulator.unique_external_outputs[name] = max(
                accumulator.unique_external_outputs.get(name, 0),
                int(elements),
            )
            accumulator.unique_external_output_bytes[name] = max(
                accumulator.unique_external_output_bytes.get(name, 0.0),
                bytes_,
            )
        return model_elems, model_bytes

    @staticmethod
    def _classify_layer_outputs(
        data: LayerData,
        memory: MemoryElements,
        byte_counts: MemoryBytes,
        topology: GraphTopology,
        declared_outputs: set[str],
        accumulator: AnalysisAccumulator,
        *,
        external_writes: Mapping[tuple[str, int], ExternalSlotWrite]
        | None = None,
    ) -> OutputIo:
        intermediate_flags = [
            any(
                consumer not in topology.transparent_layer_ids
                or topology.has_real_consumer(consumer)
                for consumer in topology.tensor_consumers.get(name) or set()
            )
            for name in data.output_names
        ]
        intermediate_elems = sum(
            value
            for value, intermediate in zip(
                memory.writes, intermediate_flags, strict=True
            )
            if intermediate
        )
        intermediate_bytes = sum(
            value
            for value, intermediate in zip(
                byte_counts.output_bytes,
                intermediate_flags,
                strict=True,
            )
            if intermediate
        )
        if declared_outputs and external_writes is not None:
            model_elems, model_bytes = (
                LayerAccountant._record_resolved_output_writes(
                    data, external_writes, accumulator
                )
            )
        else:
            external_flags = (
                [str(name) in declared_outputs for name in data.output_names]
                if declared_outputs
                else [not intermediate for intermediate in intermediate_flags]
            )
            model_elems, model_bytes = (
                LayerAccountant._record_direct_output_writes(
                    data, memory, byte_counts, external_flags, accumulator
                )
            )
        return OutputIo(
            intermediate_elems=intermediate_elems,
            model_elems=model_elems,
            intermediate_bytes=intermediate_bytes,
            model_bytes=model_bytes,
            is_intermediate=any(intermediate_flags),
        )

    def _classify_layer_io(
        self,
        data: LayerData,
        memory: MemoryElements,
        byte_counts: MemoryBytes,
        topology: GraphTopology,
        prepared: PreparedAnalysis,
        accumulator: AnalysisAccumulator,
        *,
        external_writes: Mapping[tuple[str, int], ExternalSlotWrite]
        | None = None,
    ) -> LayerIo:
        inputs = self._classify_layer_inputs(
            data,
            memory,
            byte_counts,
            topology,
            accumulator,
        )
        outputs = self._classify_layer_outputs(
            data,
            memory,
            byte_counts,
            topology,
            prepared.declared_graph_outputs,
            accumulator,
            external_writes=external_writes,
        )
        return LayerIo(
            intermediate_elems=inputs.intermediate_elems
            + outputs.intermediate_elems,
            intermediate_bytes=inputs.intermediate_bytes
            + outputs.intermediate_bytes,
            model_elems=inputs.model_elems + outputs.model_elems,
            model_bytes=inputs.model_bytes + outputs.model_bytes,
            input_is_intermediate=inputs.intermediate_elems > 0,
            output_is_intermediate=outputs.is_intermediate,
        )

    @staticmethod
    def _serialize_analyzed_layer(
        data: LayerData,
        compute: LayerCompute,
        memory: MemoryElements,
        byte_counts: MemoryBytes,
        resources: ResourceAccounting,
        io: LayerIo,
    ) -> AnalyzedLayer:
        payload: dict[str, Any] = {
            "type": data.op_type,
            "semantic_op": dict(layer_operation(data.layer)),
            "einsum_equation": data.equation,
            "is_real_einsum": compute.is_real_einsum,
            "macs": compute.macs,
            "other_ops": memory.other_ops,
            "flops": compute.flops,
            "compute_precision": (
                resources.compute_precision if compute.macs else None
            ),
            "resources": resources.resources,
            "unfused_elements": byte_counts.unfused_elems,
            "unfused_bytes": byte_counts.unfused_bytes,
            "orojenesis_elements": None,
            "fused_elements": int(io.model_elems),
            "fused_bytes": float(io.model_bytes),
            **_serialized_tensor_metadata(data, memory, byte_counts),
            "input_elements": byte_counts.input_elems,
            "output_elements": byte_counts.output_elems,
            "intermediate_elements": io.intermediate_elems,
            "intermediate_bytes": io.intermediate_bytes,
            "model_io_elements": io.model_elems,
            "model_io_bytes": io.model_bytes,
            "input_is_intermediate": io.input_is_intermediate,
            "output_is_intermediate": io.output_is_intermediate,
            "is_orphaned": memory.orphaned,
            "connections": {
                "inputs": data.input_layer_ids,
                "outputs": data.output_layer_ids,
            },
        }
        return AnalyzedLayer(
            payload=payload,
            macs=compute.macs,
            other_ops=memory.other_ops,
            flops=compute.flops,
            unfused_elems=byte_counts.unfused_elems,
            unfused_bytes=byte_counts.unfused_bytes,
            intermediate_elems=io.intermediate_elems,
            intermediate_bytes=io.intermediate_bytes,
        )

    def _intrinsic_layer(
        self,
        layer_id: str,
        layer: dict[str, Any],
        topology: GraphTopology,
        prepared: PreparedAnalysis,
    ) -> IntrinsicLayer:
        data = self._parse_layer(layer_id, layer)
        compute = self._compute_layer(data, strict=prepared.strict)
        memory = self._memory_elements(data, compute, topology)
        byte_counts = self._memory_bytes(
            data,
            memory,
            prepared.fallback_precision,
        )
        return IntrinsicLayer(
            data=data, compute=compute, memory=memory, byte_counts=byte_counts
        )

    def _account_intrinsic_layer(
        self,
        intrinsic: IntrinsicLayer,
        topology: GraphTopology,
        prepared: PreparedAnalysis,
        accumulator: AnalysisAccumulator,
        *,
        external_writes: Mapping[tuple[str, int], ExternalSlotWrite]
        | None = None,
    ) -> None:
        data, compute = intrinsic.data, intrinsic.compute
        memory, byte_counts = intrinsic.memory, intrinsic.byte_counts
        if memory.orphaned:
            accumulator.orphaned_layers.add(data.layer_id)
        accumulator.used_dtype_fallback |= byte_counts.used_dtype_fallback
        resources = self._account_layer_resources(
            data,
            compute,
            topology,
            prepared,
            accumulator,
            orphaned=memory.orphaned,
        )
        io = self._classify_layer_io(
            data,
            memory,
            byte_counts,
            topology,
            prepared,
            accumulator,
            external_writes=external_writes,
        )
        accumulator.record(
            data.layer_id,
            self._serialize_analyzed_layer(
                data,
                compute,
                memory,
                byte_counts,
                resources,
                io,
            ),
        )

    @staticmethod
    def _graph_io_totals(accumulator: AnalysisAccumulator) -> GraphIoTotals:
        fused_elements = int(
            sum(accumulator.unique_external_inputs.values())
            + sum(accumulator.unique_external_outputs.values()),
        )
        fused_bytes = float(
            sum(accumulator.unique_external_input_bytes.values())
            + sum(accumulator.unique_external_output_bytes.values()),
        )
        return GraphIoTotals(
            fused_elements=fused_elements,
            fused_bytes=fused_bytes,
            model_io_elements=sum(
                layer.get("model_io_elements", 0)
                for layer in accumulator.layers.values()
            ),
            model_io_bytes=float(
                sum(
                    layer.get("model_io_bytes", 0)
                    for layer in accumulator.layers.values()
                ),
            ),
        )

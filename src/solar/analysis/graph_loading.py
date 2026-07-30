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

from pathlib import Path
from typing import Any

import yaml

from solar.analysis.graph_context import (
    AnalysisJob,
    GraphTopology,
    PreparedAnalysis,
    build_graph_topology,
    product,
)
from solar.analysis.graph_models import (
    LayerCompute,
    LayerData,
    MemoryBytes,
    MemoryElements,
)
from solar.analysis.graph_rules import (
    SCATTER_OPS,
    SLICE_VIEW_OPS,
    ZERO_COMPUTE_OPS,
    ZERO_COPY_VIEW_OPS,
)
from solar.analysis.graph_validation import (
    validate_graph_semantics,
)
from solar.analysis.mixin_contract import AnalysisMixinContract
from solar.analysis.resources import (
    ResourceClassificationError,
    is_mfma_operation,
    mandatory_mfma_macs,
)
from solar.ir.contracts import (
    OPERATION_KIND,
    layer_contraction_analysis,
    layer_operation,
)
from solar.precision import (
    BYTES_PER_ELEMENT,
    dtype_bytes,
    normalize_dtype,
)
from solar.rocm.architecture import ArchitectureProfile
from solar.types import TensorShapes


class GraphLoadingMixin(AnalysisMixinContract):
    """Load, validate, and normalize analysis graph inputs."""

    def _resolve_analysis_paths(
        self,
        job: AnalysisJob,
    ) -> tuple[Path, Path] | None:
        source = Path(job.graph_path)
        reordered = source.parent / f"{source.stem}_reordered.yaml"
        if reordered.exists():
            if self.debug:
                print(f"Debug: using reordered graph {reordered}")
            source = reordered
        output_dir = Path(job.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if source.exists():
            return source, output_dir
        if self.debug:
            print(f"Debug: IR graph not found: {source}")
        return None

    def _load_graph(self, source: Path) -> dict[str, Any] | None:
        try:
            with open(source) as file:
                return yaml.safe_load(file) or {}
        except (OSError, UnicodeError, yaml.YAMLError) as exc:
            if self.debug:
                print(f"Debug: failed reading IR graph: {exc}")
            return None

    def _copy_source_graph(
        self,
        source: Path,
        output_dir: Path,
        *,
        enabled: bool,
    ) -> None:
        if not enabled:
            return
        try:
            destination = output_dir / source.name
            if source.resolve() != destination.resolve():
                destination.write_text(source.read_text())
        except (OSError, UnicodeError):
            if self.debug:
                print(f"Debug: failed to copy {source.name}")

    @staticmethod
    def _validate_strict_layers(all_layers: dict[str, Any]) -> None:
        failures: list[str] = []
        for layer_id, layer in all_layers.items():
            layer_type = str(layer.get("type", "")).lower()
            if layer_type != "start":
                analysis = layer_contraction_analysis(layer)
                if not analysis.is_supported:
                    failures.append(f"{layer_id}: unsupported operation")
                if (
                    layer_operation(layer).get("kind") == "einsum"
                    and not analysis.equation
                ):
                    failures.append(f"{layer_id}: empty einsum equation")
            shapes = layer.get("tensor_shapes") or {}
            dtypes = layer.get("tensor_dtypes") or {}
            for side in ("inputs", "outputs"):
                if len(shapes.get(side) or []) != len(dtypes.get(side) or []):
                    failures.append(
                        f"{layer_id}: missing explicit {side} dtype metadata",
                    )
        if failures:
            raise ValueError(
                "strict analysis refused an untrusted graph:\n- "
                + "\n- ".join(failures),
            )

    @staticmethod
    def _resolve_architecture_profile(
        architecture: str | Path | ArchitectureProfile | None,
    ) -> ArchitectureProfile | None:
        if isinstance(architecture, ArchitectureProfile):
            return architecture
        return (
            ArchitectureProfile.load(architecture)
            if architecture is not None
            else None
        )

    @staticmethod
    def _validate_profile_dtypes(
        all_layers: dict[str, Any],
        profile: ArchitectureProfile,
        fallback_precision: str,
    ) -> None:
        for layer_id, layer in all_layers.items():
            dtypes = layer.get("tensor_dtypes") or {}
            for dtype in [
                *(dtypes.get("inputs") or []),
                *(dtypes.get("outputs") or []),
            ]:
                try:
                    profile.tensor_precision(dtype, fallback_precision)
                except ValueError as exc:
                    raise ValueError(
                        f"layer {layer_id} uses an architecture-incompatible "
                        f"tensor dtype: {dtype}",
                    ) from exc

    def _prepare_analysis(self, job: AnalysisJob) -> PreparedAnalysis | None:
        paths = self._resolve_analysis_paths(job)
        if paths is None:
            return None
        source, output_dir = paths
        graph = self._load_graph(source)
        if graph is None:
            return None
        semantic_graph, semantic_complete = validate_graph_semantics(
            graph,
            strict=job.strict,
            validator=self.validator,
        )
        self._copy_source_graph(source, output_dir, enabled=job.copy_graph)
        all_layers: dict[str, Any] = graph.get("layers") or {}
        if job.strict:
            self._validate_strict_layers(all_layers)
        requested_precision = normalize_dtype(job.precision)
        profile = self._resolve_architecture_profile(job.architecture)
        if job.strict and profile is not None:
            self._validate_profile_dtypes(
                all_layers,
                profile,
                requested_precision,
            )
        return PreparedAnalysis(
            source=source,
            output_dir=output_dir,
            graph=graph,
            all_layers=all_layers,
            declared_graph_outputs=set(
                graph.get("outputs")
                or (graph.get("graph_signature") or {}).get("joint_outputs")
                or [],
            ),
            semantic_graph=semantic_graph,
            semantic_complete=semantic_complete,
            strict=job.strict,
            requested_precision=requested_precision,
            fallback_precision=requested_precision,
            element_size=BYTES_PER_ELEMENT[requested_precision],
            profile=profile,
        )

    def _build_graph_topology(
        self,
        all_layers: dict[str, Any],
        declared_graph_outputs: set[str] | None = None,
    ) -> GraphTopology:
        topology = build_graph_topology(
            all_layers,
            declared_graph_outputs,
        )
        self._debug_topology(topology)
        return topology

    def _debug_topology(self, topology: GraphTopology) -> None:
        if not self.debug:
            return
        print(f"Debug: Filtered out {len(topology.start_node_ids)} start nodes")
        if topology.bool_start_node_ids:
            print(
                f"Debug: Found {len(topology.bool_start_node_ids)} bool-typed "
                f"start nodes: {topology.bool_start_node_ids}",
            )
        print(f"Debug: Analyzing {len(topology.layers)} computation nodes")
        print(
            f"Debug: Found {len(topology.intermediate_tensors)} intermediate tensors",
        )
        for tensor_name in sorted(topology.intermediate_tensors)[:10]:
            print(f"  - {tensor_name}")
        if topology.transparent_layer_ids:
            print(
                f"Debug: {len(topology.transparent_layer_ids)} transparent view layers",
            )
        if topology.bool_layers:
            print(
                f"Debug: Skipping memory for {len(topology.bool_layers)} "
                f"bool-derived layers: {sorted(topology.bool_layers)}",
            )

    @staticmethod
    def _parse_layer(layer_id: str, layer: dict[str, Any]) -> LayerData:
        op_type = str(layer.get("type", "unknown"))
        analysis = layer_contraction_analysis(layer)
        shapes = layer.get("tensor_shapes") or {}
        types = layer.get("tensor_types") or {}
        dtypes = layer.get("tensor_dtypes") or {}
        names = layer.get("tensor_names") or {}
        connections = layer.get("connections") or {}
        input_shapes = list(shapes.get("inputs") or [])
        output_shapes = list(shapes.get("outputs") or [])
        return LayerData(
            layer_id=layer_id,
            layer=layer,
            op_type=op_type,
            equation=analysis.equation,
            is_real_einsum=analysis.is_contraction,
            input_layer_ids=list(connections.get("inputs") or []),
            output_layer_ids=list(connections.get("outputs") or []),
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            input_types=list(types.get("inputs") or []),
            output_types=list(types.get("outputs") or []),
            input_dtypes=list(dtypes.get("inputs") or []),
            output_dtypes=list(dtypes.get("outputs") or []),
            input_names=list(names.get("inputs") or []),
            output_names=list(names.get("outputs") or []),
            input_sizes=[
                product(shape) if isinstance(shape, list) else 0
                for shape in input_shapes
            ],
            output_sizes=[
                product(shape) if isinstance(shape, list) else 0
                for shape in output_shapes
            ],
        )

    def _compute_layer(self, data: LayerData, *, strict: bool) -> LayerCompute:
        shapes = TensorShapes(
            inputs=data.input_shapes,
            outputs=data.output_shapes,
        )
        operation = data.op_type
        if operation == "addmm" and len(data.input_shapes) >= 3:
            operation = "matmul"
            shapes = TensorShapes(
                inputs=data.input_shapes[1:3],
                outputs=data.output_shapes,
            )
        semantic_kind = str(layer_operation(data.layer).get("kind", ""))
        semantic = layer_operation(data.layer)
        semantic_target = str(semantic.get("target", data.op_type))
        try:
            if semantic_kind == OPERATION_KIND:
                cost = mandatory_mfma_macs(
                    semantic_target,
                    data.input_shapes,
                    data.output_shapes,
                    semantic,
                )
            elif data.is_real_einsum and data.equation:
                cost = int(
                    self.einsum_analyzer.get_compute_cost(
                        operation,
                        shapes,
                        equation=data.equation,
                    ),
                )
            else:
                cost = int(
                    self.einsum_analyzer.get_compute_cost(operation, shapes),
                )
        except Exception as exc:
            if strict:
                raise ResourceClassificationError(
                    f"operation handler failed for {data.op_type!r}: {exc}"
                ) from exc
            cost = 0
        is_real_einsum = data.is_real_einsum
        if data.op_type in ZERO_COMPUTE_OPS:
            cost = 0
            is_real_einsum = False
        contraction = is_real_einsum or is_mfma_operation(semantic_target)
        macs = cost if contraction else 0
        return LayerCompute(
            is_real_einsum=is_real_einsum,
            macs=macs,
            other_ops=0 if contraction else cost,
            flops=2 * macs,
        )

    @staticmethod
    def _scatter_write_elements(data: LayerData) -> int:
        if len(data.input_sizes) >= 2:
            return max(sorted(data.input_sizes)[:-1])
        if data.input_sizes:
            return min(data.input_sizes)
        return min(data.output_sizes) if data.output_sizes else 0

    @staticmethod
    def _memory_elements(
        data: LayerData,
        compute: LayerCompute,
        topology: GraphTopology,
    ) -> MemoryElements:
        reads = list(data.input_sizes)
        writes = list(data.output_sizes)
        other_ops = compute.other_ops
        if data.op_type in {"embedding", "embedding_bag"}:
            reads = [0] * len(data.input_sizes)
            if data.input_sizes:
                reads[-1] = min(sum(data.input_sizes), sum(data.output_sizes))
            writes = [0] * len(data.output_sizes)
            other_ops = 0
        if data.op_type in ZERO_COPY_VIEW_OPS:
            reads = [0] * len(data.input_sizes)
            writes = [0] * len(data.output_sizes)
            other_ops = 0
        elif data.op_type in SLICE_VIEW_OPS:
            reads = [sum(data.output_sizes)] if data.input_sizes else []
            reads += [0] * max(0, len(data.input_sizes) - 1)
            writes = [0] * len(data.output_sizes)
            other_ops = 0
        elif data.op_type in SCATTER_OPS:
            slice_elements = GraphLoadingMixin._scatter_write_elements(data)
            reads = [0] * len(data.input_sizes)
            writes = [slice_elements] if data.output_sizes else []
            writes += [0] * max(0, len(data.output_sizes) - 1)
            other_ops = 0
        orphaned = data.layer_id not in topology.live_layer_ids
        if data.layer_id in topology.dead_end_layers and data.input_layer_ids:
            orphaned |= all(
                topology.source_is_orphan(item) for item in data.input_layer_ids
            )
            if (
                not orphaned
                and data.op_type in SCATTER_OPS
                and topology.source_is_orphan(data.input_layer_ids[0])
            ):
                orphaned = True
        if orphaned:
            reads = [0] * len(data.input_sizes)
            writes = [0] * len(data.output_sizes)
            other_ops = 0
        return MemoryElements(reads, writes, other_ops, orphaned)

    @staticmethod
    def _memory_bytes(
        data: LayerData,
        memory: MemoryElements,
        fallback_precision: str,
    ) -> MemoryBytes:
        input_elems = int(sum(memory.reads))
        output_elems = int(sum(memory.writes))
        used_fallback = any(
            count > 0 and index >= len(data.input_dtypes)
            for index, count in enumerate(memory.reads)
        ) or any(
            count > 0 and index >= len(data.output_dtypes)
            for index, count in enumerate(memory.writes)
        )
        input_bytes = [
            float(count)
            * dtype_bytes(
                data.input_dtypes[index]
                if index < len(data.input_dtypes)
                else None,
                fallback_precision,
            )
            for index, count in enumerate(memory.reads)
        ]
        output_bytes = [
            float(count)
            * dtype_bytes(
                data.output_dtypes[index]
                if index < len(data.output_dtypes)
                else None,
                fallback_precision,
            )
            for index, count in enumerate(memory.writes)
        ]
        return MemoryBytes(
            input_elems=input_elems,
            output_elems=output_elems,
            unfused_elems=input_elems + output_elems,
            input_bytes=input_bytes,
            output_bytes=output_bytes,
            unfused_bytes=float(sum(input_bytes) + sum(output_bytes)),
            used_dtype_fallback=used_fallback,
        )

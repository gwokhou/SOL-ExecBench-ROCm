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

"""Analyze an einsum graph into hardware-independent metrics.

This module implements the **second stage** of the Solar pipeline:

  `einsum_graph.yaml`  ->  `analysis.yaml`

The output `analysis.yaml` is intended to be hardware-independent and includes:
- per-layer: macs, flops (= 2 * macs), tensor dtypes, and exact byte traffic
- totals across the graph

Memory access models (elements are diagnostic; byte totals use each tensor dtype):
- ``unfused``: every per-operation input and output access;
- ``fused``: compulsory, deduplicated graph-external I/O;
- ``prefetched`` / ``io_lower_bound``: compulsory I/O plus the safely composable
  excess traffic selected from a capacity-constrained Orojenesis curve.

Formal schema-v3 analysis also emits conservative fusion regions, hierarchy
capacity pressure, the pinned solver inputs/raw curve and their SHA-256 hashes,
and separate compute/memory overlap components. Verified canonical and extended
MatMul regions use pinned multi-einsum FFMT evidence; unsupported composition
is never approximated into a scored bound.

Note: input_elements includes all inputs to an operation (including weights/biases).
Weights are treated as inputs since they are just another operand to the computation.

Note: "start" nodes are filtered out before analysis as they represent model inputs,
not actual computation. Their outputs are treated as external inputs to the graph.

See SOL_GUIDE.md for detailed explanation of the three SOL models.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set, Union, cast

import yaml

from solar.einsum.analyzer import EinsumAnalyzer
from solar.einsum.semantics import (
    EINSUM_GRAPH_SCHEMA_VERSION,
    SemanticGraphError,
    validate_semantic_graph,
)
from solar.analysis.fusion import FusionPlanner
from solar.analysis.graph_rules import (
    BOOL_DTYPES,
    QUANTIZED_PAYLOAD_PASSTHROUGH,
    SCATTER_OPS,
    SLICE_VIEW_OPS,
    TRANSPARENT_OPS,
    ZERO_COMPUTE_OPS,
    ZERO_COPY_VIEW_OPS,
)
from solar.analysis.operand_provenance import (
    contraction_external_source_dtypes,
    contraction_operands_are_graph_external,
)
from solar.analysis.graph_models import (
    AnalysisAccumulator as _AnalysisAccumulator,
    AnalyzedLayer as _AnalyzedLayer,
    FormalAnalysis as _FormalAnalysis,
    FusionPlan as _FusionPlan,
    GraphIoTotals as _GraphIoTotals,
    InputIo as _InputIo,
    LayerCompute as _LayerCompute,
    LayerData as _LayerData,
    LayerIo as _LayerIo,
    LowerBound as _LowerBound,
    MemoryBytes as _MemoryBytes,
    MemoryElements as _MemoryElements,
    OutputIo as _OutputIo,
    ResourceAccounting as _ResourceAccounting,
)
from solar.analysis.orojenesis import (
    OrojenesisRunner,
    find_multi_einsum_chains,
    find_multi_einsum_regions,
    select_capacity_point,
)
from solar.analysis.resources import (
    RESOURCE_MODEL_VERSION,
    classify_layer_resources,
    merge_resource_work,
)
from solar.analysis.reporting import build_analysis_result, write_analysis
from solar.schema_versions import (
    OROJENESIS_ANALYSIS_SCHEMA_VERSION,
    SOLAR_ANALYSIS_SCHEMA_VERSION as SOLAR_ANALYSIS_SCHEMA_VERSION,
    SOLAR_REQUEST_MANIFEST_SCHEMA_VERSION as SOLAR_REQUEST_MANIFEST_SCHEMA_VERSION,
)
from solar.rocm.architecture import ArchitectureProfile, MemoryLevel
from solar.common.constants import (
    BYTES_PER_ELEMENT,
    DEFAULT_PRECISION,
    dtype_bytes,
    normalize_dtype,
)
from solar.common.types import TensorShapes
from solar.common.utils import ensure_directory

PathLike = Union[str, Path]


def _product(shape: List[int]) -> int:
    out = 1
    for d in shape:
        out *= int(d)
    return int(out)


@dataclass(frozen=True, slots=True)
class _AnalysisJob:
    graph_path: PathLike
    output_dir: PathLike
    precision: str
    copy_graph: bool
    strict: bool
    architecture: str | Path | ArchitectureProfile | None
    orojenesis_runner: OrojenesisRunner | None
    require_orojenesis: bool


@dataclass(frozen=True, slots=True)
class _PreparedAnalysis:
    source: Path
    output_dir: Path
    graph: Dict[str, Any]
    all_layers: Dict[str, Any]
    declared_graph_outputs: Set[str]
    semantic_graph: bool
    semantic_complete: bool
    strict: bool
    requested_precision: str
    fallback_precision: str
    element_size: float
    profile: ArchitectureProfile | None


@dataclass(frozen=True, slots=True)
class _GraphTopology:
    layers: Dict[str, Any]
    start_node_ids: Set[str]
    bool_start_node_ids: Set[str]
    all_layer_ids: Set[str]
    transparent_layer_ids: Set[str]
    tensor_producers: Dict[str, str]
    tensor_consumers: Dict[str, Set[str]]
    intermediate_tensors: Set[str]
    bool_layers: Set[str]
    dead_end_layers: Set[str]

    def trace_source_through_views(self, layer_id: str) -> str:
        """Trace backward through transparent view layers to the real source."""
        visited: Set[str] = set()
        current = layer_id
        while current in self.transparent_layer_ids and current not in visited:
            visited.add(current)
            connections = (self.layers[current].get("connections") or {}).get(
                "inputs"
            ) or []
            if not connections:
                break
            current = connections[0]
        return current

    def has_real_consumer(self, layer_id: str) -> bool:
        """Return whether an output reaches a non-transparent graph layer."""
        visited: Set[str] = set()
        queue = [layer_id]
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            connections = (self.layers.get(current, {}).get("connections") or {}).get(
                "outputs"
            ) or []
            for output_id in connections:
                if output_id in self.transparent_layer_ids:
                    queue.append(output_id)
                elif output_id in self.all_layer_ids:
                    return True
        return False

    def source_is_orphan(self, connection_id: str) -> bool:
        source = (
            self.trace_source_through_views(connection_id)
            if connection_id in self.transparent_layer_ids
            else connection_id
        )
        return source not in self.all_layer_ids and source not in self.start_node_ids

    def dequantized_payload_precision(
        self,
        tensor_name: str,
        profile: ArchitectureProfile | None,
        fallback_precision: str,
        visited: Set[str] | None = None,
    ) -> str | None:
        """Trace one contraction payload to an exact low-precision cast."""
        if (
            profile is None
            or (producer_id := self.tensor_producers.get(tensor_name)) is None
        ):
            return None
        seen = set(visited or ())
        if producer_id in seen:
            return None
        seen.add(producer_id)
        producer = self.layers[producer_id]
        semantic = producer.get("semantic_op") or {}
        target = str(semantic.get("target", ""))
        names = (producer.get("tensor_names") or {}).get("inputs") or []
        dtypes = producer.get("tensor_dtypes") or {}
        inputs = list(dtypes.get("inputs") or [])
        outputs = list(dtypes.get("outputs") or [])
        if target == "to" and inputs and outputs:
            source = profile.tensor_precision(inputs[0], fallback_precision)
            destination = profile.tensor_precision(outputs[0], fallback_precision)
            return (
                source
                if source in {"fp8", "int8", "int4"} and source != destination
                else None
            )
        if target in QUANTIZED_PAYLOAD_PASSTHROUGH and names:
            return self.dequantized_payload_precision(
                str(names[0]), profile, fallback_precision, seen
            )
        if target == "mul":
            candidates = {
                candidate
                for name in names
                if (
                    candidate := self.dequantized_payload_precision(
                        str(name), profile, fallback_precision, seen
                    )
                )
                is not None
            }
            if len(candidates) == 1:
                return candidates.pop()
        return None


class EinsumGraphAnalyzer:
    """Analyze `einsum_graph.yaml` and write `analysis.yaml`."""

    def __init__(self, debug: bool = False) -> None:
        self.debug = debug
        self.einsum_analyzer = EinsumAnalyzer(debug=debug)

    def analyze_graph(
        self,
        einsum_graph_path: PathLike,
        output_dir: PathLike,
        *,
        precision: str = DEFAULT_PRECISION,
        copy_graph: bool = True,
        strict: bool = False,
        architecture: str | Path | ArchitectureProfile | None = None,
        orojenesis_runner: OrojenesisRunner | None = None,
        require_orojenesis: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Analyze an einsum graph and write `analysis.yaml`.

        Args:
            einsum_graph_path: Path to `einsum_graph.yaml`.
            output_dir: Directory to write `analysis.yaml` into.
            precision: Tensor precision for byte calculations (e.g., fp32, bf16).
            copy_graph: If True, copy the einsum graph into output dir using the
                canonical name `einsum_graph.yaml`.
            strict: Reject unsupported layers and every implicit dtype fallback.

        Returns:
            Analysis dict, or None on failure.
        """
        return self._analyze_job(
            _AnalysisJob(
                graph_path=einsum_graph_path,
                output_dir=output_dir,
                precision=precision,
                copy_graph=copy_graph,
                strict=strict,
                architecture=architecture,
                orojenesis_runner=orojenesis_runner,
                require_orojenesis=require_orojenesis,
            )
        )

    def _resolve_analysis_paths(self, job: _AnalysisJob) -> tuple[Path, Path] | None:
        source = Path(job.graph_path)
        reordered = source.parent / "einsum_graph_reordered.yaml"
        if source.name == "einsum_graph.yaml" and reordered.exists():
            if self.debug:
                print(f"Debug: using reordered graph {reordered}")
            source = reordered
        output_dir = ensure_directory(job.output_dir)
        if source.exists():
            return source, output_dir
        if self.debug:
            print(f"Debug: einsum graph not found: {source}")
        return None

    def _load_graph(self, source: Path) -> Dict[str, Any] | None:
        try:
            with open(source) as file:
                return yaml.safe_load(file) or {}
        except Exception as exc:
            if self.debug:
                print(f"Debug: failed reading einsum graph: {exc}")
            return None

    @staticmethod
    def _validate_graph_semantics(
        graph: Dict[str, Any], strict: bool
    ) -> tuple[bool, bool]:
        semantic_graph = (
            int(graph.get("schema_version", 0)) == EINSUM_GRAPH_SCHEMA_VERSION
        )
        if not semantic_graph:
            raise ValueError(
                "analysis requires executable semantics: einsum graph must use "
                f"schema_version={EINSUM_GRAPH_SCHEMA_VERSION}"
            )
        try:
            validate_semantic_graph(graph)
        except SemanticGraphError as exc:
            if strict:
                raise ValueError(
                    f"strict analysis requires executable semantics: {exc}"
                ) from exc
            return True, False
        return True, True

    def _copy_source_graph(
        self, source: Path, output_dir: Path, *, enabled: bool
    ) -> None:
        if not enabled:
            return
        try:
            destination = output_dir / "einsum_graph.yaml"
            if source.resolve() != destination.resolve():
                destination.write_text(source.read_text())
        except Exception:
            if self.debug:
                print("Debug: failed to copy einsum_graph.yaml")

    @staticmethod
    def _validate_strict_layers(all_layers: Dict[str, Any]) -> None:
        failures: List[str] = []
        for layer_id, layer in all_layers.items():
            layer_type = str(layer.get("type", "")).lower()
            if layer_type != "start":
                if layer.get("is_einsum_supportable") is not True:
                    failures.append(f"{layer_id}: unsupported operation")
                semantic = layer.get("semantic_op") or {}
                if semantic.get("kind") == "einsum" and not layer.get(
                    "einsum_equation"
                ):
                    failures.append(f"{layer_id}: empty einsum equation")
            shapes = layer.get("tensor_shapes") or {}
            dtypes = layer.get("tensor_dtypes") or {}
            for side in ("inputs", "outputs"):
                if len(shapes.get(side) or []) != len(dtypes.get(side) or []):
                    failures.append(
                        f"{layer_id}: missing explicit {side} dtype metadata"
                    )
        if failures:
            raise ValueError(
                "strict analysis refused an untrusted graph:\n- "
                + "\n- ".join(failures)
            )

    @staticmethod
    def _resolve_architecture_profile(
        architecture: str | Path | ArchitectureProfile | None,
    ) -> ArchitectureProfile | None:
        if isinstance(architecture, ArchitectureProfile):
            return architecture
        return (
            ArchitectureProfile.load(architecture) if architecture is not None else None
        )

    @staticmethod
    def _validate_profile_dtypes(
        all_layers: Dict[str, Any],
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
                        f"tensor dtype: {dtype}"
                    ) from exc

    def _prepare_analysis(self, job: _AnalysisJob) -> _PreparedAnalysis | None:
        paths = self._resolve_analysis_paths(job)
        if paths is None:
            return None
        source, output_dir = paths
        graph = self._load_graph(source)
        if graph is None:
            return None
        semantic_graph, semantic_complete = self._validate_graph_semantics(
            graph, job.strict
        )
        self._copy_source_graph(source, output_dir, enabled=job.copy_graph)
        all_layers: Dict[str, Any] = graph.get("layers") or {}
        if job.strict:
            self._validate_strict_layers(all_layers)
        requested_precision = normalize_dtype(job.precision)
        profile = self._resolve_architecture_profile(job.architecture)
        if job.strict and profile is not None:
            self._validate_profile_dtypes(all_layers, profile, requested_precision)
        return _PreparedAnalysis(
            source=source,
            output_dir=output_dir,
            graph=graph,
            all_layers=all_layers,
            declared_graph_outputs=set(
                graph.get("outputs")
                or (graph.get("graph_signature") or {}).get("joint_outputs")
                or []
            ),
            semantic_graph=semantic_graph,
            semantic_complete=semantic_complete,
            strict=job.strict,
            requested_precision=requested_precision,
            fallback_precision=requested_precision,
            element_size=BYTES_PER_ELEMENT[requested_precision],
            profile=profile,
        )

    @staticmethod
    def _partition_graph_layers(
        all_layers: Dict[str, Any],
    ) -> tuple[Dict[str, Any], Set[str], Set[str]]:
        layers: Dict[str, Any] = {}
        start_node_ids: Set[str] = set()
        bool_start_node_ids: Set[str] = set()
        for layer_id, layer in all_layers.items():
            if str(layer.get("type", "")).lower() != "start":
                layers[layer_id] = layer
                continue
            start_node_ids.add(layer_id)
            output_dtypes = (layer.get("tensor_dtypes") or {}).get("outputs") or []
            if output_dtypes and all(
                str(dtype) in BOOL_DTYPES for dtype in output_dtypes
            ):
                bool_start_node_ids.add(layer_id)
        return layers, start_node_ids, bool_start_node_ids

    @staticmethod
    def _build_tensor_indices(
        layers: Dict[str, Any],
    ) -> tuple[Dict[str, str], Dict[str, Set[str]]]:
        producers: Dict[str, str] = {}
        consumers: Dict[str, Set[str]] = {}
        for layer_id, layer in layers.items():
            names = layer.get("tensor_names") or {}
            for output_name in names.get("outputs") or []:
                producers[output_name] = layer_id
        for layer_id, layer in layers.items():
            names = layer.get("tensor_names") or {}
            for input_name in names.get("inputs") or []:
                if input_name in producers:
                    consumers.setdefault(input_name, set()).add(layer_id)
        return producers, consumers

    @staticmethod
    def _propagate_bool_layers(
        layers: Dict[str, Any], bool_start_node_ids: Set[str]
    ) -> Set[str]:
        bool_layers: Set[str] = set()
        changed = bool(bool_start_node_ids)
        while changed:
            changed = False
            for layer_id, layer in layers.items():
                if layer_id in bool_layers:
                    continue
                input_ids = list((layer.get("connections") or {}).get("inputs") or [])
                if input_ids and all(
                    item in bool_start_node_ids or item in bool_layers
                    for item in input_ids
                ):
                    bool_layers.add(layer_id)
                    changed = True
        return bool_layers

    def _build_graph_topology(self, all_layers: Dict[str, Any]) -> _GraphTopology:
        layers, start_ids, bool_start_ids = self._partition_graph_layers(all_layers)
        all_layer_ids = set(layers)
        transparent_ids = {
            layer_id
            for layer_id, layer in layers.items()
            if str(layer.get("type", "")).lower() in TRANSPARENT_OPS
        }
        producers, consumers = self._build_tensor_indices(layers)
        intermediate_tensors = {name for name in producers if bool(consumers.get(name))}
        topology = _GraphTopology(
            layers=layers,
            start_node_ids=start_ids,
            bool_start_node_ids=bool_start_ids,
            all_layer_ids=all_layer_ids,
            transparent_layer_ids=transparent_ids,
            tensor_producers=producers,
            tensor_consumers=consumers,
            intermediate_tensors=intermediate_tensors,
            bool_layers=self._propagate_bool_layers(layers, bool_start_ids),
            dead_end_layers=set(),
        )
        topology = replace(
            topology,
            dead_end_layers={
                layer_id
                for layer_id in layers
                if not topology.has_real_consumer(layer_id)
            },
        )
        self._debug_topology(topology)
        return topology

    def _debug_topology(self, topology: _GraphTopology) -> None:
        if not self.debug:
            return
        print(f"Debug: Filtered out {len(topology.start_node_ids)} start nodes")
        if topology.bool_start_node_ids:
            print(
                f"Debug: Found {len(topology.bool_start_node_ids)} bool-typed "
                f"start nodes: {topology.bool_start_node_ids}"
            )
        print(f"Debug: Analyzing {len(topology.layers)} computation nodes")
        print(f"Debug: Found {len(topology.intermediate_tensors)} intermediate tensors")
        for tensor_name in sorted(topology.intermediate_tensors)[:10]:
            print(f"  - {tensor_name}")
        if topology.transparent_layer_ids:
            print(
                f"Debug: {len(topology.transparent_layer_ids)} transparent view layers"
            )
        if topology.bool_layers:
            print(
                f"Debug: Skipping memory for {len(topology.bool_layers)} "
                f"bool-derived layers: {sorted(topology.bool_layers)}"
            )

    @staticmethod
    def _parse_layer(layer_id: str, layer: Dict[str, Any]) -> _LayerData:
        op_type = str(layer.get("type", "unknown"))
        if "is_real_einsum" not in layer:
            raise ValueError(
                f"Layer '{layer_id}' (type={op_type}) is missing "
                "'is_real_einsum' field. All layers in the einsum graph must "
                "specify is_real_einsum: true/false."
            )
        shapes = layer.get("tensor_shapes") or {}
        types = layer.get("tensor_types") or {}
        dtypes = layer.get("tensor_dtypes") or {}
        names = layer.get("tensor_names") or {}
        connections = layer.get("connections") or {}
        input_shapes = list(shapes.get("inputs") or [])
        output_shapes = list(shapes.get("outputs") or [])
        return _LayerData(
            layer_id=layer_id,
            layer=layer,
            op_type=op_type,
            equation=str(layer.get("einsum_equation", "") or ""),
            is_real_einsum=bool(layer["is_real_einsum"]),
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
                _product(shape) if isinstance(shape, list) else 0
                for shape in input_shapes
            ],
            output_sizes=[
                _product(shape) if isinstance(shape, list) else 0
                for shape in output_shapes
            ],
        )

    def _compute_layer(self, data: _LayerData) -> _LayerCompute:
        shapes = TensorShapes(inputs=data.input_shapes, outputs=data.output_shapes)
        try:
            if data.is_real_einsum and data.equation:
                cost = int(
                    self.einsum_analyzer.get_compute_cost(
                        data.op_type, shapes, equation=data.equation
                    )
                )
            else:
                cost = int(self.einsum_analyzer.get_compute_cost(data.op_type, shapes))
        except Exception:
            cost = 0
        is_real_einsum = data.is_real_einsum
        if data.op_type in ZERO_COMPUTE_OPS:
            cost = 0
            is_real_einsum = False
        macs = cost if is_real_einsum else 0
        return _LayerCompute(
            is_real_einsum=is_real_einsum,
            macs=macs,
            other_ops=0 if is_real_einsum else cost,
            flops=2 * macs,
        )

    @staticmethod
    def _scatter_write_elements(data: _LayerData) -> int:
        if len(data.input_sizes) >= 2:
            return max(sorted(data.input_sizes)[:-1])
        if data.input_sizes:
            return min(data.input_sizes)
        return min(data.output_sizes) if data.output_sizes else 0

    @staticmethod
    def _memory_elements(
        data: _LayerData,
        compute: _LayerCompute,
        topology: _GraphTopology,
    ) -> _MemoryElements:
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
            slice_elements = EinsumGraphAnalyzer._scatter_write_elements(data)
            reads = [0] * len(data.input_sizes)
            writes = [slice_elements] if data.output_sizes else []
            writes += [0] * max(0, len(data.output_sizes) - 1)
            other_ops = 0
        orphaned = False
        if data.layer_id in topology.dead_end_layers and data.input_layer_ids:
            orphaned = all(
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
        return _MemoryElements(reads, writes, other_ops, orphaned)

    @staticmethod
    def _memory_bytes(
        data: _LayerData,
        memory: _MemoryElements,
        fallback_precision: str,
    ) -> _MemoryBytes:
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
                data.input_dtypes[index] if index < len(data.input_dtypes) else None,
                fallback_precision,
            )
            for index, count in enumerate(memory.reads)
        ]
        output_bytes = [
            float(count)
            * dtype_bytes(
                data.output_dtypes[index] if index < len(data.output_dtypes) else None,
                fallback_precision,
            )
            for index, count in enumerate(memory.writes)
        ]
        return _MemoryBytes(
            input_elems=input_elems,
            output_elems=output_elems,
            unfused_elems=input_elems + output_elems,
            input_bytes=input_bytes,
            output_bytes=output_bytes,
            unfused_bytes=float(sum(input_bytes) + sum(output_bytes)),
            used_dtype_fallback=used_fallback,
        )

    def _account_layer_resources(
        self,
        data: _LayerData,
        compute: _LayerCompute,
        topology: _GraphTopology,
        prepared: _PreparedAnalysis,
        accumulator: _AnalysisAccumulator,
        *,
        orphaned: bool,
    ) -> _ResourceAccounting:
        compute_precisions = [
            normalized
            for dtype in data.input_dtypes
            if (normalized := normalize_dtype(dtype, prepared.fallback_precision))
            in {
                "fp64",
                "fp32",
                "tf32",
                "bf16",
                "fp16",
                "fp8",
                "nvfp4",
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
        semantic = data.layer.get("semantic_op") or {}
        if compute.macs and semantic.get("kind") == "einsum":
            payload_precisions = [
                topology.dequantized_payload_precision(
                    name, prepared.profile, prepared.fallback_precision
                )
                for name in data.input_names
            ]
            if (
                len(payload_precisions) >= 2
                and all(payload_precisions)
                and len(set(payload_precisions)) == 1
            ):
                compute_precision = str(payload_precisions[0])
                resource_precision = compute_precision
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
                "model_version": RESOURCE_MODEL_VERSION,
                "work": {},
                "classification": "exempt",
                "exemption_reason": "orphaned_dead_end",
                "formulas": [],
            }
        layer_work = resources.get("work")
        if not isinstance(layer_work, Mapping) or any(
            not isinstance(value, Mapping) for value in layer_work.values()
        ):
            raise TypeError(f"layer {data.layer_id!r} resource work is not a mapping")
        merge_resource_work(
            accumulator.resource_work,
            cast(Mapping[str, Mapping[str, Any]], layer_work),
        )
        accumulator.resource_coverage[str(resources["classification"])] += 1
        return _ResourceAccounting(compute_precision, resources)

    @staticmethod
    def _classify_layer_inputs(
        data: _LayerData,
        memory: _MemoryElements,
        byte_counts: _MemoryBytes,
        topology: _GraphTopology,
        accumulator: _AnalysisAccumulator,
    ) -> _InputIo:
        intermediate_elems = 0
        model_elems = 0
        intermediate_bytes = 0.0
        model_bytes = 0.0
        for index, memory_read in enumerate(memory.reads):
            if memory_read <= 0:
                continue
            input_type = (
                data.input_types[index] if index < len(data.input_types) else "weight"
            )
            input_name = (
                data.input_names[index] if index < len(data.input_names) else ""
            )
            graph_internal = False
            if input_type != "weight" and input_name in topology.tensor_producers:
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
                    accumulator.unique_external_input_bytes.get(input_name, 0.0),
                    byte_counts.input_bytes[index],
                )
        return _InputIo(
            intermediate_elems, model_elems, intermediate_bytes, model_bytes
        )

    @staticmethod
    def _classify_layer_outputs(
        data: _LayerData,
        memory: _MemoryElements,
        byte_counts: _MemoryBytes,
        topology: _GraphTopology,
        declared_outputs: Set[str],
        accumulator: _AnalysisAccumulator,
    ) -> _OutputIo:
        intermediate_flags = [
            any(
                consumer not in topology.transparent_layer_ids
                or topology.has_real_consumer(consumer)
                for consumer in topology.tensor_consumers.get(name) or set()
            )
            for name in data.output_names
        ]
        external_flags = [
            str(name) in declared_outputs or not intermediate
            for name, intermediate in zip(data.output_names, intermediate_flags)
        ]
        intermediate_elems = sum(
            value
            for value, intermediate in zip(memory.writes, intermediate_flags)
            if intermediate
        )
        intermediate_bytes = sum(
            value
            for value, intermediate in zip(byte_counts.output_bytes, intermediate_flags)
            if intermediate
        )
        model_elems = sum(
            value for value, external in zip(memory.writes, external_flags) if external
        )
        model_bytes = sum(
            value
            for value, external in zip(byte_counts.output_bytes, external_flags)
            if external
        )
        for index, (name, external) in enumerate(
            zip(data.output_names, external_flags)
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
                accumulator.unique_external_outputs.get(name, 0), int(elements)
            )
            accumulator.unique_external_output_bytes[name] = max(
                accumulator.unique_external_output_bytes.get(name, 0.0), bytes_
            )
        return _OutputIo(
            intermediate_elems,
            model_elems,
            intermediate_bytes,
            model_bytes,
            any(intermediate_flags),
        )

    def _classify_layer_io(
        self,
        data: _LayerData,
        memory: _MemoryElements,
        byte_counts: _MemoryBytes,
        topology: _GraphTopology,
        prepared: _PreparedAnalysis,
        accumulator: _AnalysisAccumulator,
    ) -> _LayerIo:
        inputs = self._classify_layer_inputs(
            data, memory, byte_counts, topology, accumulator
        )
        outputs = self._classify_layer_outputs(
            data,
            memory,
            byte_counts,
            topology,
            prepared.declared_graph_outputs,
            accumulator,
        )
        return _LayerIo(
            intermediate_elems=inputs.intermediate_elems + outputs.intermediate_elems,
            intermediate_bytes=inputs.intermediate_bytes + outputs.intermediate_bytes,
            model_elems=inputs.model_elems + outputs.model_elems,
            model_bytes=inputs.model_bytes + outputs.model_bytes,
            input_is_intermediate=inputs.intermediate_elems > 0,
            output_is_intermediate=outputs.is_intermediate,
        )

    @staticmethod
    def _serialize_analyzed_layer(
        data: _LayerData,
        compute: _LayerCompute,
        memory: _MemoryElements,
        byte_counts: _MemoryBytes,
        resources: _ResourceAccounting,
        io: _LayerIo,
    ) -> _AnalyzedLayer:
        payload: Dict[str, Any] = {
            "type": data.op_type,
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
            "memory_elements": {"inputs": memory.reads, "outputs": memory.writes},
            "memory_bytes": {
                "inputs": byte_counts.input_bytes,
                "outputs": byte_counts.output_bytes,
            },
            "tensor_dtypes": {
                "inputs": data.input_dtypes,
                "outputs": data.output_dtypes,
            },
            "tensor_types": {
                "inputs": data.input_types,
                "outputs": data.output_types,
            },
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
        return _AnalyzedLayer(
            payload=payload,
            macs=compute.macs,
            other_ops=memory.other_ops,
            flops=compute.flops,
            unfused_elems=byte_counts.unfused_elems,
            unfused_bytes=byte_counts.unfused_bytes,
            intermediate_elems=io.intermediate_elems,
            intermediate_bytes=io.intermediate_bytes,
        )

    def _analyze_layer(
        self,
        layer_id: str,
        layer: Dict[str, Any],
        topology: _GraphTopology,
        prepared: _PreparedAnalysis,
        accumulator: _AnalysisAccumulator,
    ) -> None:
        data = self._parse_layer(layer_id, layer)
        compute = self._compute_layer(data)
        memory = self._memory_elements(data, compute, topology)
        if memory.orphaned:
            accumulator.orphaned_layers.add(layer_id)
        byte_counts = self._memory_bytes(data, memory, prepared.fallback_precision)
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
            data, memory, byte_counts, topology, prepared, accumulator
        )
        accumulator.record(
            layer_id,
            self._serialize_analyzed_layer(
                data, compute, memory, byte_counts, resources, io
            ),
        )

    @staticmethod
    def _graph_io_totals(accumulator: _AnalysisAccumulator) -> _GraphIoTotals:
        fused_elements = int(
            sum(accumulator.unique_external_inputs.values())
            + sum(accumulator.unique_external_outputs.values())
        )
        fused_bytes = float(
            sum(accumulator.unique_external_input_bytes.values())
            + sum(accumulator.unique_external_output_bytes.values())
        )
        return _GraphIoTotals(
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
                )
            ),
        )

    @staticmethod
    def _plan_fusion(
        prepared: _PreparedAnalysis, topology: _GraphTopology
    ) -> _FusionPlan:
        chains = find_multi_einsum_chains(prepared.all_layers)
        regions = find_multi_einsum_regions(prepared.all_layers)
        region_paths = [
            path for region in regions for path in region.get("physical_paths") or []
        ]
        verified_views = {
            str(layer_id)
            for region in regions
            for path in region.get("physical_paths") or []
            for layer_id in path
            if str(
                (
                    prepared.all_layers.get(str(layer_id), {}).get("semantic_op") or {}
                ).get("target", "")
            )
            in {"view", "transpose", "permute", "squeeze", "unsqueeze"}
        }
        hierarchy = (
            prepared.profile.memory_hierarchy if prepared.profile is not None else ()
        )
        return _FusionPlan(
            fusion=FusionPlanner(
                prepared.graph,
                multi_einsum_chains=[*chains, *region_paths],
                verified_view_nodes=sorted(verified_views),
            ).plan(hierarchy),
            chains=chains,
            regions=regions,
            einsum_layers={
                layer_id: layer
                for layer_id, layer in topology.layers.items()
                if (layer.get("semantic_op") or {}).get("kind") == "einsum"
            },
        )

    @staticmethod
    def _last_cache(profile: ArchitectureProfile | None) -> MemoryLevel | None:
        if profile is None:
            return None
        known = [
            level
            for level in profile.memory_hierarchy
            if level.capacity_bytes is not None and level.name != "vram"
        ]
        if not known:
            return None
        return max(
            known,
            key=lambda level: int(level.capacity_bytes or 0),
        )

    @staticmethod
    def _word_bits(dtypes: List[str], element_size: float) -> int:
        return min(
            (int(dtype_bytes(dtype) * 8) for dtype in dtypes),
            default=int(element_size * 8),
        )

    @staticmethod
    def _select_capacity_and_rewrite_evidence(
        result: Dict[str, Any],
        last_cache: MemoryLevel | None,
        require_orojenesis: bool,
        missing_point_error: str,
        evidence_root: Path,
    ) -> None:
        if last_cache is not None:
            point = select_capacity_point(
                result["curve"], int(last_cache.capacity_bytes or 0)
            )
            if point is None and require_orojenesis:
                raise ValueError(missing_point_error)
            result["selected_capacity"] = {
                "level": last_cache.name,
                "capacity_bytes": last_cache.capacity_bytes,
                "point": point,
            }
        for evidence in result.get("evidence_files", {}).values():
            evidence["path"] = str(evidence_root / str(evidence["path"]))

    def _run_chain_evidence(
        self,
        plan: _FusionPlan,
        runner: OrojenesisRunner,
        prepared: _PreparedAnalysis,
        last_cache: MemoryLevel | None,
        orojenesis: Dict[str, Any],
        require_orojenesis: bool,
    ) -> None:
        for index, layer_ids in enumerate(plan.chains):
            layers = [
                (layer_id, plan.einsum_layers[layer_id]) for layer_id in layer_ids
            ]
            dtypes = [
                str(dtype)
                for _, layer in layers
                for side in ("inputs", "outputs")
                for dtype in ((layer.get("tensor_dtypes") or {}).get(side) or [])
            ]
            chain_id = f"chain_{index}"
            result = runner.run_multi_chain(
                layers,
                prepared.output_dir / "orojenesis" / "chains" / chain_id,
                word_bits=self._word_bits(dtypes, prepared.element_size),
            )
            self._select_capacity_and_rewrite_evidence(
                result,
                last_cache,
                require_orojenesis,
                "multi-einsum Orojenesis produced no point within "
                f"{last_cache.name if last_cache else '<cache>'} capacity for {chain_id}",
                Path("orojenesis") / "chains" / chain_id,
            )
            orojenesis["chains"][chain_id] = result

    def _run_region_evidence(
        self,
        plan: _FusionPlan,
        runner: OrojenesisRunner,
        prepared: _PreparedAnalysis,
        last_cache: MemoryLevel | None,
        orojenesis: Dict[str, Any],
        require_orojenesis: bool,
    ) -> None:
        for index, problem in enumerate(plan.regions):
            region_id = f"region_{index}"
            layer_ids = [str(item) for item in problem.get("schedule") or []]
            dtypes = [
                str(dtype)
                for layer_id in layer_ids
                for side in ("inputs", "outputs")
                for dtype in (
                    (plan.einsum_layers[layer_id].get("tensor_dtypes") or {}).get(side)
                    or []
                )
            ]
            result = runner.run_multi_region(
                problem,
                prepared.output_dir / "orojenesis" / "regions" / region_id,
                word_bits=self._word_bits(dtypes, prepared.element_size),
            )
            self._select_capacity_and_rewrite_evidence(
                result,
                last_cache,
                require_orojenesis,
                "multi-einsum region produced no point within "
                f"{last_cache.name if last_cache else '<cache>'} capacity for {region_id}",
                Path("orojenesis") / "regions" / region_id,
            )
            orojenesis["regions"][region_id] = result

    def _run_layer_evidence(
        self,
        plan: _FusionPlan,
        runner: OrojenesisRunner,
        prepared: _PreparedAnalysis,
        last_cache: MemoryLevel | None,
        orojenesis: Dict[str, Any],
        require_orojenesis: bool,
    ) -> None:
        multi_member_ids = {layer_id for chain in plan.chains for layer_id in chain}
        multi_member_ids.update(
            str(layer_id)
            for region in plan.regions
            for layer_id in region.get("schedule") or []
        )
        for layer_id, layer in plan.einsum_layers.items():
            if layer_id in multi_member_ids:
                continue
            tensor_dtypes = layer.get("tensor_dtypes") or {}
            dtypes = [
                *(
                    str(dtype)
                    for side in ("inputs", "outputs")
                    for dtype in tensor_dtypes.get(side) or []
                ),
                *contraction_external_source_dtypes(layer, prepared.all_layers),
            ]
            result = runner.run_layer(
                layer,
                prepared.output_dir / "orojenesis" / layer_id,
                word_bits=self._word_bits(dtypes, prepared.element_size),
            )
            self._select_capacity_and_rewrite_evidence(
                result,
                last_cache,
                require_orojenesis,
                f"Orojenesis produced no point within "
                f"{last_cache.name if last_cache else '<cache>'} capacity for {layer_id}",
                Path("orojenesis") / layer_id,
            )
            orojenesis["layers"][layer_id] = result

    def _run_orojenesis_evidence(
        self,
        plan: _FusionPlan,
        runner: OrojenesisRunner,
        prepared: _PreparedAnalysis,
        orojenesis: Dict[str, Any],
        *,
        require_orojenesis: bool,
    ) -> None:
        orojenesis["status"] = "complete"
        orojenesis["toolchain"] = getattr(runner, "toolchain_identity", None)
        if orojenesis["toolchain"] is None and require_orojenesis:
            raise ValueError(
                "strict formal analysis requires Orojenesis toolchain identity"
            )
        last_cache = self._last_cache(prepared.profile)
        self._run_chain_evidence(
            plan,
            runner,
            prepared,
            last_cache,
            orojenesis,
            require_orojenesis,
        )
        self._run_region_evidence(
            plan,
            runner,
            prepared,
            last_cache,
            orojenesis,
            require_orojenesis,
        )
        self._run_layer_evidence(
            plan,
            runner,
            prepared,
            last_cache,
            orojenesis,
            require_orojenesis,
        )

    @staticmethod
    def _audit_layer_evidence(
        plan: _FusionPlan,
        orojenesis: Dict[str, Any],
        region_by_layer: Dict[str, Any],
        all_layers: Dict[str, Any],
    ) -> List[float]:
        excesses: List[float] = []
        for layer_id, result in orojenesis["layers"].items():
            layer = plan.einsum_layers[layer_id]
            point = (result.get("selected_capacity") or {}).get("point")
            region = region_by_layer[layer_id]
            external = contraction_operands_are_graph_external(layer, all_layers)
            applicable = bool(point and external)
            result["formal_applicability"] = {
                "applicable": applicable,
                "region": region["id"],
                "graph_input_operands": external,
                "operand_provenance": (
                    "graph_input_or_recomputable_preprocess" if external else "internal"
                ),
                "reason": (
                    "graph_input_or_recomputable_preprocess_contraction"
                    if applicable
                    else "internal_operand_requires_multi_einsum_composition"
                ),
            }
            if not applicable:
                continue
            assert point is not None
            word_bytes = int(result["word_bits"]) // 8
            names = layer.get("tensor_names") or {}
            shapes = layer.get("tensor_shapes") or {}
            modeled_tensors = {
                str(name): list(shape)
                for side in ("inputs", "outputs")
                for name, shape in zip(names.get(side) or [], shapes.get(side) or [])
            }
            compulsory_bytes = float(
                sum(_product(shape) for shape in modeled_tensors.values()) * word_bytes
            )
            solver_bytes = float(point["dram_bytes"])
            result["audited_dram_bytes"] = solver_bytes
            result["modeled_compulsory_bytes"] = compulsory_bytes
            excesses.append(max(0.0, solver_bytes - compulsory_bytes))
        return excesses

    @staticmethod
    def _audit_chain_evidence(
        plan: _FusionPlan,
        orojenesis: Dict[str, Any],
        region_by_layer: Dict[str, Any],
    ) -> List[float]:
        excesses: List[float] = []
        for result in orojenesis["chains"].values():
            point = (result.get("selected_capacity") or {}).get("point")
            descriptors = ((result.get("problem") or {}).get("chain") or {}).get(
                "layers"
            ) or []
            layer_ids = [str(item.get("id")) for item in descriptors]
            region_ids = {
                str(region_by_layer[layer_id]["id"])
                for layer_id in layer_ids
                if layer_id in region_by_layer
            }
            applicable = bool(
                point
                and len(layer_ids) >= 2
                and len(region_ids) == 1
                and all(layer_id in plan.einsum_layers for layer_id in layer_ids)
            )
            result["formal_applicability"] = {
                "applicable": applicable,
                "region": next(iter(region_ids), None),
                "layer_ids": layer_ids,
                "operand_provenance": "graph_inputs_and_internal_chain_edges",
                "reason": (
                    "verified_linear_matmul_tiled_fusion"
                    if applicable
                    else "multi_einsum_chain_or_region_mismatch"
                ),
            }
            if not applicable:
                continue
            assert point is not None
            first, last = descriptors[0], descriptors[-1]
            compulsory_elements = int(first["m"]) * int(first["k"])
            compulsory_elements += sum(
                int(item["k"]) * int(item["n"]) for item in descriptors
            )
            compulsory_elements += int(last["m"]) * int(last["n"])
            compulsory_bytes = float(
                compulsory_elements * (int(result["word_bits"]) // 8)
            )
            solver_bytes = float(point["dram_bytes"])
            result["audited_dram_bytes"] = solver_bytes
            result["modeled_compulsory_bytes"] = compulsory_bytes
            excesses.append(max(0.0, solver_bytes - compulsory_bytes))
        return excesses

    @staticmethod
    def _audit_region_evidence(
        plan: _FusionPlan,
        orojenesis: Dict[str, Any],
        region_by_layer: Dict[str, Any],
    ) -> List[float]:
        excesses: List[float] = []
        for result in orojenesis["regions"].values():
            point = (result.get("selected_capacity") or {}).get("point")
            problem = result.get("problem") or {}
            descriptors = problem.get("nodes") or []
            layer_ids = [str(item.get("id")) for item in descriptors]
            region_ids = {
                str(region_by_layer[layer_id]["id"])
                for layer_id in layer_ids
                if layer_id in region_by_layer
            }
            applicable = bool(
                point
                and len(layer_ids) >= 2
                and len(region_ids) == 1
                and all(layer_id in plan.einsum_layers for layer_id in layer_ids)
            )
            result["formal_applicability"] = {
                "applicable": applicable,
                "region": next(iter(region_ids), None),
                "layer_ids": layer_ids,
                "operand_provenance": (
                    "graph_inputs_and_verified_internal_region_edges"
                ),
                "reason": (
                    "verified_matmul_region_tiled_fusion"
                    if applicable
                    else "multi_einsum_region_or_fusion_mismatch"
                ),
            }
            if not applicable:
                continue
            assert point is not None
            by_id = {str(item["id"]): item for item in descriptors}
            roots = [str(item) for item in problem.get("roots") or []]
            leaves = [str(item) for item in problem.get("leaves") or []]
            compulsory_elements = sum(
                int(by_id[root]["m"]) * int(by_id[root]["k"]) for root in roots
            )
            compulsory_elements += sum(
                int(item["k"]) * int(item["n"]) for item in descriptors
            )
            compulsory_elements += sum(
                int(by_id[leaf]["m"]) * int(by_id[leaf]["n"]) for leaf in leaves
            )
            compulsory_bytes = float(
                compulsory_elements * (int(result["word_bits"]) // 8)
            )
            solver_bytes = float(point["dram_bytes"])
            result["audited_dram_bytes"] = solver_bytes
            result["modeled_compulsory_bytes"] = compulsory_bytes
            excesses.append(max(0.0, solver_bytes - compulsory_bytes))
        return excesses

    @staticmethod
    def _formal_coverage(plan: _FusionPlan, orojenesis: Dict[str, Any]) -> int:
        applicable_layers = sum(
            bool((result.get("formal_applicability") or {}).get("applicable"))
            for result in orojenesis["layers"].values()
        )
        for category in ("chains", "regions"):
            applicable_layers += sum(
                len((result.get("formal_applicability") or {}).get("layer_ids") or [])
                for result in orojenesis[category].values()
                if (result.get("formal_applicability") or {}).get("applicable")
            )
        orojenesis["formal_coverage"] = {
            "applicable_layers": applicable_layers,
            "total_layers": len(plan.einsum_layers),
        }
        return applicable_layers

    def _audit_orojenesis_evidence(
        self,
        plan: _FusionPlan,
        orojenesis: Dict[str, Any],
        prepared: _PreparedAnalysis,
        audited_fused_bytes: float,
    ) -> tuple[float, bool]:
        region_by_layer = {
            layer_id: region
            for region in plan.fusion["regions"]
            for layer_id in region["layers"]
        }
        excesses = self._audit_layer_evidence(
            plan, orojenesis, region_by_layer, prepared.all_layers
        )
        excesses.extend(self._audit_chain_evidence(plan, orojenesis, region_by_layer))
        excesses.extend(self._audit_region_evidence(plan, orojenesis, region_by_layer))
        self._formal_coverage(plan, orojenesis)
        results = [
            *orojenesis["layers"].values(),
            *orojenesis["chains"].values(),
            *orojenesis["regions"].values(),
        ]
        formal_bound = bool(excesses) and all(
            (result.get("selected_capacity") or {}).get("point") is not None
            for result in results
        )
        return audited_fused_bytes + max(excesses, default=0.0), formal_bound

    def _run_formal_analysis(
        self,
        prepared: _PreparedAnalysis,
        topology: _GraphTopology,
        io_totals: _GraphIoTotals,
        runner: OrojenesisRunner | None,
        *,
        require_orojenesis: bool,
    ) -> _FormalAnalysis:
        orojenesis: Dict[str, Any] = {
            "schema_version": OROJENESIS_ANALYSIS_SCHEMA_VERSION,
            "status": (
                "not_applicable" if not prepared.semantic_graph else "not_requested"
            ),
            "toolchain": None,
            "layers": {},
            "chains": {},
            "regions": {},
        }
        if not (prepared.semantic_graph and prepared.semantic_complete):
            return _FormalAnalysis(
                None,
                orojenesis,
                io_totals.fused_bytes,
                io_totals.fused_bytes,
                False,
            )
        plan = self._plan_fusion(prepared, topology)
        if plan.einsum_layers and runner is None and require_orojenesis:
            raise ValueError(
                "strict formal analysis requires the pinned Orojenesis toolchain"
            )
        if runner is not None and plan.einsum_layers:
            self._run_orojenesis_evidence(
                plan,
                runner,
                prepared,
                orojenesis,
                require_orojenesis=require_orojenesis,
            )
        elif not plan.einsum_layers:
            orojenesis["status"] = "not_applicable"
        audited_prefetched_bytes = io_totals.fused_bytes
        formal_bound = not plan.einsum_layers
        if plan.einsum_layers and orojenesis["status"] == "complete":
            audited_prefetched_bytes, formal_bound = self._audit_orojenesis_evidence(
                plan,
                orojenesis,
                prepared,
                io_totals.fused_bytes,
            )
        return _FormalAnalysis(
            fusion=plan.fusion,
            orojenesis=orojenesis,
            audited_fused_bytes=io_totals.fused_bytes,
            audited_prefetched_bytes=audited_prefetched_bytes,
            formal_bound=formal_bound,
        )

    @staticmethod
    def _lower_bound(
        prepared: _PreparedAnalysis,
        accumulator: _AnalysisAccumulator,
        formal: _FormalAnalysis,
        *,
        require_orojenesis: bool,
    ) -> _LowerBound:
        seconds: float | None = None
        resource_seconds: dict[str, float] = {}
        compute_resource: str | None = None
        components: Dict[str, Any] | None = None
        if (
            prepared.profile is not None
            and prepared.semantic_graph
            and prepared.semantic_complete
        ):
            resource_seconds = prepared.profile.resource_seconds(
                accumulator.resource_work
            )
            compute_seconds = max(resource_seconds.values(), default=0.0)
            if resource_seconds:
                compute_resource = max(
                    sorted(resource_seconds), key=resource_seconds.__getitem__
                )
            fused_memory_seconds = (
                formal.audited_fused_bytes
                / prepared.profile.memory_bandwidth_bytes_per_second
            )
            prefetched_memory_seconds = (
                formal.audited_prefetched_bytes
                / prepared.profile.memory_bandwidth_bytes_per_second
            )
            seconds = max(compute_seconds, prefetched_memory_seconds)
            components = {
                "compute_seconds": compute_seconds,
                "resource_seconds": resource_seconds,
                "compute_resource": compute_resource,
                "fused_memory_seconds": fused_memory_seconds,
                "fused_unoverlapped_seconds": compute_seconds + fused_memory_seconds,
                "prefetched_memory_seconds": prefetched_memory_seconds,
                "prefetched_overlapped_seconds": seconds,
            }
        if require_orojenesis and not formal.formal_bound:
            raise ValueError(
                "formal analysis did not produce a complete tile-aware bound"
            )
        return _LowerBound(seconds, resource_seconds, compute_resource, components)

    def _analyze_job(self, job: _AnalysisJob) -> Optional[Dict[str, Any]]:
        """Run validated graph analysis through explicit accounting stages."""
        prepared = self._prepare_analysis(job)
        if prepared is None:
            return None
        topology = self._build_graph_topology(prepared.all_layers)
        accumulator = _AnalysisAccumulator()
        for layer_id, layer in topology.layers.items():
            self._analyze_layer(layer_id, layer, topology, prepared, accumulator)
        io_totals = self._graph_io_totals(accumulator)
        formal = self._run_formal_analysis(
            prepared,
            topology,
            io_totals,
            job.orojenesis_runner,
            require_orojenesis=job.require_orojenesis,
        )
        lower_bound = self._lower_bound(
            prepared,
            accumulator,
            formal,
            require_orojenesis=job.require_orojenesis,
        )
        analysis = build_analysis_result(
            prepared,
            accumulator,
            io_totals,
            formal,
            lower_bound,
            start_node_count=len(topology.start_node_ids),
            intermediate_tensor_count=len(topology.intermediate_tensors),
        )
        output_path = write_analysis(prepared.output_dir, analysis)
        if self.debug:
            print(f"✅ Wrote analysis: {output_path}")
        return analysis

    # Maps metadata orig_dtypes keywords to Solar precision names
    _QUANT_DTYPE_MAP = {
        "nvfp4": "nvfp4",
        "float4_e2m1fn_x2": "nvfp4",
        "fp8": "fp8",
        "float8_e4m3fn": "fp8",
        "float8_e5m2": "fp8",
        "float8_e4m3fnuz": "fp8",
        "float8_e5m2fnuz": "fp8",
    }

    def _resolve_quant_precision(self, einsum_graph_path: Path) -> Optional[str]:
        """Search for metadata.yaml near the einsum graph and return quant precision.

        Walks up from the einsum_graph_path looking for metadata.yaml
        (max 3 levels). Picks highest-throughput quant dtype (nvfp4 > fp8).
        """
        search_dir = einsum_graph_path.parent
        for _ in range(3):
            candidate = search_dir / "metadata.yaml"
            if candidate.exists():
                try:
                    with open(candidate) as f:
                        meta = yaml.safe_load(f) or {}
                except Exception:
                    return None

                best = None
                for conv in meta.get("dtype_conversions") or []:
                    orig = str(conv.get("orig_dtypes", "")).lower()
                    for keyword, prec in self._QUANT_DTYPE_MAP.items():
                        if keyword in orig:
                            if best is None or BYTES_PER_ELEMENT.get(
                                prec, 99
                            ) < BYTES_PER_ELEMENT.get(best, 99):
                                best = prec
                            break
                return best
            search_dir = search_dir.parent
        return None


__all__ = ["EinsumGraphAnalyzer"]

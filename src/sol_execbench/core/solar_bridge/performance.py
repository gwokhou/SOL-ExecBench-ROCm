# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Validated SOLAR-to-performance-model boundary."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, cast

from sol_execbench.core.bench.performance_model.models import (
    CompositeGraphDescriptor,
    CompositeGraphEdge,
    CompositeGraphNode,
    CrossEntropyDescriptor,
    CrossEntropyReduction,
    ElementwiseDescriptor,
    ElementwiseOperationClass,
    EvidenceReference,
    FusionRegion,
    IndexedReadDescriptor,
    IndexedReadOperation,
    IndexedUpdateDescriptor,
    IndexedUpdateOperation,
    MatmulDescriptor,
    ReductionDescriptor,
    ReductionOperation,
    SemanticCharacterization,
    SoftmaxDescriptor,
    SoftmaxOperation,
    TensorDType,
    TransposeDescriptor,
    UnsupportedDescriptor,
    WorkloadKind,
)
from sol_execbench.core.data.definition_models import DType
from sol_execbench.core.integrity import (
    sha256_file,
    validate_relative_artifact_path,
)
from sol_execbench.core.solar_bridge.models import (
    SolarRequestManifest,
)
from solar.artifacts import load_yaml_artifact
from solar.schema_versions import (
    SOLAR_ANALYSIS_SCHEMA_VERSION,
    SOLAR_REQUEST_MANIFEST_SCHEMA_VERSION,
)

_SIMPLE_OPS = frozenset(
    {"abs", "add", "clamp", "div", "mul", "neg", "relu", "sub"}
)
_TRANSCENDENTAL_OPS = frozenset({"cos", "exp", "log", "sigmoid", "sin", "tanh"})
_COMPOSITE_OPS = frozenset({"gelu"})
_ELEMENTWISE_TYPES = _SIMPLE_OPS | _TRANSCENDENTAL_OPS | _COMPOSITE_OPS
_TRANSPOSE_TYPES = frozenset({"permute", "transpose"})
_REDUCTION_TYPES = {
    "sum": ReductionOperation.SUM,
    "mean": ReductionOperation.MEAN,
    "rms_norm": ReductionOperation.RMS_NORM,
    "layer_norm": ReductionOperation.LAYER_NORM,
    "native_layer_norm": ReductionOperation.LAYER_NORM,
}
_SOFTMAX_TYPES = {
    "softmax": SoftmaxOperation.SOFTMAX,
    "_softmax": SoftmaxOperation.SOFTMAX,
    "log_softmax": SoftmaxOperation.LOG_SOFTMAX,
    "_log_softmax": SoftmaxOperation.LOG_SOFTMAX,
}
_INDEXED_READ_TYPES = {
    "gather": IndexedReadOperation.GATHER,
    "index_select": IndexedReadOperation.INDEX_SELECT,
    "embedding": IndexedReadOperation.EMBEDDING,
}
_INDEXED_UPDATE_TYPES = {
    "scatter": IndexedUpdateOperation.SCATTER,
    "index_copy": IndexedUpdateOperation.INDEX_COPY,
    "index_put": IndexedUpdateOperation.INDEX_PUT,
    "scatter_add": IndexedUpdateOperation.SCATTER_ADD,
    "index_add": IndexedUpdateOperation.INDEX_ADD,
}
_MATMUL_TYPES = frozenset({"bmm", "gemm", "matmul", "mm"})
_IGNORED_TYPES = frozenset(
    {
        "",
        "start",
        "input",
        "clone",
        "contiguous",
        "expand",
        "getitem",
        "identity",
        "reshape",
        "squeeze",
        "unsqueeze",
        "view",
    }
)
_MAX_COMPOSITE_NODES = 32


def load_manifest_semantic_characterization(
    manifest_path: str | Path,
    *,
    workload_uuid: str,
    definition: str,
) -> SemanticCharacterization:
    """Verify a SOLAR manifest and load its cited analysis."""
    path = Path(manifest_path)
    manifest = SolarRequestManifest.from_yaml(
        path.read_text(encoding="utf-8"),
    )
    if manifest.analysis_id != f"{definition}:{workload_uuid}":
        raise ValueError("solar_manifest_workload_identity_mismatch")
    if manifest.sol_score_eligible is not True:
        raise ValueError("solar_manifest_bound_not_eligible")
    analysis_path, digest = _manifest_analysis(path.parent, manifest)
    return load_semantic_characterization(
        analysis_path,
        workload_uuid=workload_uuid,
        expected_sha256=digest,
    )


def _manifest_analysis(
    root: Path,
    manifest: SolarRequestManifest,
) -> tuple[Path, str]:
    matches = [
        item
        for item in manifest.artifacts
        if item.path == "solar-analysis.yaml"
    ]
    if len(matches) != 1:
        raise ValueError("solar_manifest_analysis_reference_missing")
    record = matches[0]
    relative = validate_relative_artifact_path(record.path)
    digest = record.sha256
    analysis_path = root.resolve() / relative
    if (
        analysis_path.is_symlink()
        or not analysis_path.is_file()
        or sha256_file(analysis_path) != digest
    ):
        raise ValueError("solar_manifest_analysis_sha256_mismatch")
    return analysis_path, digest


def load_semantic_characterization(
    path: str | Path,
    *,
    workload_uuid: str,
    expected_sha256: str | None = None,
) -> SemanticCharacterization:
    """Load and strictly characterize one canonical analysis."""
    source_path = Path(path)
    actual_sha256 = sha256_file(source_path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError("solar_analysis_sha256_mismatch")
    data = load_yaml_artifact(source_path).data
    if data.get("schema_version") != SOLAR_ANALYSIS_SCHEMA_VERSION:
        raise ValueError("unsupported_solar_analysis_schema")
    layers = _mapping(data.get("layers"), "layers")
    total = _mapping(data.get("total"), "total")
    metadata = _mapping(data.get("metadata"), "metadata")
    descriptor, kind, reasons = _semantic_descriptor(layers, metadata)
    regions = _fusion_regions(metadata.get("fusion"))
    if not regions:
        reasons.append("fusion_regions_missing")
    elif not _fusion_regions_cover_operations(layers, regions):
        reasons.append("fusion_region_identity_mismatch")
    return SemanticCharacterization(
        workload_uuid=workload_uuid,
        workload_kind=kind,
        descriptor=descriptor,
        resource_work=_resource_work(total.get("resource_work")),
        fusion_regions=regions,
        semantic_flops=_nonnegative_number(total.get("flops"), "total.flops"),
        semantic_bytes=_semantic_bytes(total),
        t_sol_ms=(
            _nonnegative_number(
                total.get("lower_bound_seconds"),
                "total.lower_bound_seconds",
            )
            * 1_000.0
        ),
        source=EvidenceReference(
            kind="solar_analysis",
            path=str(source_path),
            sha256=actual_sha256,
        ),
        reason_codes=reasons,
    )


def _semantic_descriptor(
    layers: Mapping[str, object],
    metadata: Mapping[str, object],
) -> tuple[object, WorkloadKind, list[str]]:
    operation_layers = _operation_layers(layers)
    operation_types = [item[1] for item in operation_layers]
    special = _special_graph_descriptor(layers, operation_layers, metadata)
    if special is not None:
        return special
    if operation_types and set(operation_types) <= _ELEMENTWISE_TYPES:
        descriptor = _elementwise_descriptor(operation_layers)
        if descriptor is not None:
            return descriptor, WorkloadKind.ELEMENTWISE, []
    if len(operation_layers) == 1 and operation_types[0] in _TRANSPOSE_TYPES:
        descriptor = _transpose_descriptor(operation_layers[0][0])
        if descriptor is not None:
            return descriptor, WorkloadKind.TRANSPOSE, []
    if len(operation_layers) == 1 and operation_types[0] in _REDUCTION_TYPES:
        descriptor = _reduction_descriptor(
            operation_layers[0][0],
            _REDUCTION_TYPES[operation_types[0]],
        )
        if descriptor is not None:
            return descriptor, WorkloadKind.REDUCTION, []
    if len(operation_layers) == 1 and operation_types[0] in _MATMUL_TYPES:
        descriptor = _matmul_descriptor(
            operation_layers[0][0],
            batched=operation_types[0] == "bmm",
        )
        if descriptor is not None:
            return descriptor, WorkloadKind.MATMUL, []
    if len(operation_layers) == 1 and operation_types[0] in _SOFTMAX_TYPES:
        descriptor = _softmax_descriptor(
            operation_layers[0][0],
            _SOFTMAX_TYPES[operation_types[0]],
        )
        if descriptor is not None:
            return descriptor, WorkloadKind.SOFTMAX, []
    decomposed_cross_entropy = _decomposed_cross_entropy_descriptor(
        operation_layers
    )
    if decomposed_cross_entropy is not None:
        return decomposed_cross_entropy, WorkloadKind.CROSS_ENTROPY, []
    if len(operation_layers) == 1 and operation_types[0] == "cross_entropy":
        descriptor = _cross_entropy_descriptor(operation_layers[0][0])
        if descriptor is not None:
            return descriptor, WorkloadKind.CROSS_ENTROPY, []
    if len(operation_layers) == 1 and operation_types[0] in _INDEXED_READ_TYPES:
        descriptor = _indexed_read_descriptor(
            operation_layers[0][0],
            _INDEXED_READ_TYPES[operation_types[0]],
        )
        if descriptor is not None:
            return descriptor, WorkloadKind.INDEXED_READ, []
    if (
        len(operation_layers) == 1
        and operation_types[0] in _INDEXED_UPDATE_TYPES
    ):
        descriptor = _indexed_update_descriptor(
            operation_layers[0][0],
            _INDEXED_UPDATE_TYPES[operation_types[0]],
        )
        if descriptor is not None:
            return descriptor, WorkloadKind.INDEXED_UPDATE, []
    composite = _composite_descriptor(layers, operation_layers, metadata)
    if composite is not None:
        kind = {
            "transformer_block": WorkloadKind.TRANSFORMER,
            "concurrent_graph": WorkloadKind.CONCURRENT,
        }.get(composite.graph_class, WorkloadKind.COMPOSITE)
        return composite, kind, []
    reasons = ["unsupported_workload_descriptor"]
    return (
        UnsupportedDescriptor(reason_codes=reasons),
        WorkloadKind.UNSUPPORTED,
        reasons,
    )


def _special_graph_descriptor(
    layers: Mapping[str, object],
    operation_layers: list[tuple[Mapping[str, object], str]],
    metadata: Mapping[str, object],
) -> tuple[object, WorkloadKind, list[str]] | None:
    special_graph = _declared_special_graph(metadata)
    if special_graph is None:
        return None
    composite = _composite_descriptor(layers, operation_layers, metadata)
    if composite is None or composite.graph_class != special_graph:
        return None
    kind = (
        WorkloadKind.TRANSFORMER
        if special_graph == "transformer_block"
        else WorkloadKind.CONCURRENT
    )
    return composite, kind, []


def _declared_special_graph(
    metadata: Mapping[str, object],
) -> Literal["transformer_block", "concurrent_graph"] | None:
    semantics = metadata.get("performance_semantics")
    if not isinstance(semantics, Mapping):
        return None
    graph_class = semantics.get("graph_class")
    if graph_class == "transformer_block":
        return "transformer_block"
    if graph_class == "concurrent_graph":
        return "concurrent_graph"
    return None


def _operation_layers(
    layers: Mapping[str, object],
) -> list[tuple[Mapping[str, object], str]]:
    result: list[tuple[Mapping[str, object], str]] = []
    for value in layers.values():
        if not isinstance(value, Mapping):
            continue
        layer = cast("Mapping[str, object]", value)
        operation = str(layer.get("type", "")).lower()
        alias_only = _is_metadata_or_alias_only(layer)
        if operation not in _IGNORED_TYPES and (
            not alias_only or operation in _TRANSPOSE_TYPES
        ):
            result.append((layer, operation))
    return result


def _is_metadata_or_alias_only(layer: Mapping[str, object]) -> bool:
    resources = layer.get("resources")
    return (
        isinstance(resources, Mapping)
        and resources.get("classification") == "exempt"
        and resources.get("exemption_reason") == "metadata_or_alias_only"
    )


def _elementwise_descriptor(
    layers: list[tuple[Mapping[str, object], str]],
) -> ElementwiseDescriptor | None:
    if any(
        not _tensor_metadata(layer, "inputs")
        or not _tensor_metadata(layer, "outputs")
        for layer, _operation in layers
    ):
        return None
    outputs = [
        metadata
        for layer, _operation in layers
        for metadata in _tensor_metadata(layer, "outputs")
    ]
    if not outputs:
        return None
    shape, dtype = max(outputs, key=lambda item: _shape_elements(item[0]))
    if dtype not in {TensorDType.FLOAT32, TensorDType.BFLOAT16}:
        return None
    tensors = [
        metadata
        for layer, _operation in layers
        for direction in ("inputs", "outputs")
        for metadata in _tensor_metadata(layer, direction)
    ]
    if not tensors or any(item[1] is not dtype for item in tensors):
        return None
    counts: dict[ElementwiseOperationClass, float] = {}
    for layer, operation in layers:
        group = _elementwise_group(operation)
        layer_outputs = _tensor_metadata(layer, "outputs")
        amount = sum(_shape_elements(item[0]) for item in layer_outputs)
        counts[group] = counts.get(group, 0.0) + float(amount)
    return ElementwiseDescriptor(
        shape=shape,
        dtype=dtype,
        operations=counts,
    )


def _elementwise_group(operation: str) -> ElementwiseOperationClass:
    if operation in _SIMPLE_OPS:
        return ElementwiseOperationClass.SIMPLE
    if operation in _TRANSCENDENTAL_OPS:
        return ElementwiseOperationClass.TRANSCENDENTAL
    return ElementwiseOperationClass.COMPOSITE


def _transpose_descriptor(
    layer: Mapping[str, object],
) -> TransposeDescriptor | None:
    inputs = _tensor_metadata(layer, "inputs")
    outputs = _tensor_metadata(layer, "outputs")
    if len(inputs) != 1 or len(outputs) != 1:
        return None
    input_shape, dtype = inputs[0]
    output_shape, output_dtype = outputs[0]
    if (
        len(input_shape) != 2
        or output_shape != [input_shape[1], input_shape[0]]
        or output_dtype is not dtype
    ):
        return None
    rows, columns = input_shape
    return TransposeDescriptor(
        rows=rows,
        columns=columns,
        dtype=dtype,
        element_bytes=_dtype_bytes(dtype),
        input_strides=(columns, 1),
        output_strides=(rows, 1),
    )


def _reduction_descriptor(
    layer: Mapping[str, object],
    operation: ReductionOperation,
) -> ReductionDescriptor | None:
    inputs = _tensor_metadata(layer, "inputs")
    outputs = _tensor_metadata(layer, "outputs")
    if not inputs or not outputs:
        return None
    input_shape, input_dtype = inputs[0]
    output_shape, output_dtype = outputs[0]
    if (
        not input_shape
        or input_dtype not in {TensorDType.BFLOAT16, TensorDType.FLOAT32}
        or output_dtype not in {TensorDType.BFLOAT16, TensorDType.FLOAT32}
    ):
        return None
    expected_outputs = (
        {tuple(input_shape)}
        if operation
        in {ReductionOperation.RMS_NORM, ReductionOperation.LAYER_NORM}
        else {
            tuple(input_shape[:-1]),
            (*input_shape[:-1], 1),
        }
    )
    if tuple(output_shape) not in expected_outputs:
        return None
    if operation is ReductionOperation.LAYER_NORM:
        normalized_shape = _attributes(layer).get("normalized_shape")
        if normalized_shape not in (
            input_shape[-1],
            (input_shape[-1],),
            [input_shape[-1]],
        ):
            return None
    return ReductionDescriptor(
        operation=operation,
        outer_rows=max(math.prod(input_shape[:-1]), 1),
        reduction_width=input_shape[-1],
        input_dtype=input_dtype,
        output_dtype=output_dtype,
    )


def _matmul_descriptor(
    layer: Mapping[str, object],
    *,
    batched: bool,
) -> MatmulDescriptor | None:
    inputs = _tensor_metadata(layer, "inputs")
    outputs = _tensor_metadata(layer, "outputs")
    if len(inputs) < 2 or len(outputs) != 1:
        return None
    left, right = inputs[0], inputs[1]
    output_shape, output_dtype = outputs[0]
    if left[1] is not right[1] or left[1] not in {
        TensorDType.FLOAT16,
        TensorDType.FLOAT32,
    }:
        return None
    if output_dtype not in {left[1], TensorDType.FLOAT32}:
        return None
    expected_rank = 3 if batched else 2
    if any(len(shape) != expected_rank for shape in (left[0], right[0])):
        return None
    batch = left[0][0] if batched else 1
    m, k = left[0][-2:]
    right_k, n = right[0][-2:]
    if (
        k != right_k
        or output_shape[-2:] != [m, n]
        or (batched and right[0][0] != batch)
        or (batched and output_shape[0] != batch)
        or (not batched and len(output_shape) != 2)
    ):
        return None
    return MatmulDescriptor(
        batch=batch,
        m=m,
        n=n,
        k=k,
        leading_dimension_a=k,
        leading_dimension_b=n,
        leading_dimension_c=n,
        input_dtype=left[1],
        output_dtype=output_dtype,
        contiguous=bool(_attributes(layer).get("contiguous", True)),
        batch_stride_a=(m * k if batched else None),
        batch_stride_b=(k * n if batched else None),
        batch_stride_c=(m * n if batched else None),
    )


def _softmax_descriptor(
    layer: Mapping[str, object],
    operation: SoftmaxOperation,
) -> SoftmaxDescriptor | None:
    inputs = _tensor_metadata(layer, "inputs")
    outputs = _tensor_metadata(layer, "outputs")
    if len(inputs) != 1 or len(outputs) != 1 or inputs[0] != outputs[0]:
        return None
    shape, dtype = inputs[0]
    if dtype not in {TensorDType.BFLOAT16, TensorDType.FLOAT32}:
        return None
    axis = _normalized_axis(_attributes(layer).get("dim"), len(shape))
    if axis != len(shape) - 1:
        return None
    return SoftmaxDescriptor(
        operation=operation,
        outer_rows=max(math.prod(shape[:-1]), 1),
        reduction_width=shape[-1],
        input_dtype=dtype,
        output_dtype=dtype,
    )


def _cross_entropy_descriptor(
    layer: Mapping[str, object],
) -> CrossEntropyDescriptor | None:
    raw = _raw_tensor_metadata(layer, "inputs")
    outputs = _raw_tensor_metadata(layer, "outputs")
    if len(raw) < 2 or len(outputs) != 1:
        return None
    logits_shape, logits_dtype = raw[0]
    target_shape, target_dtype = raw[1]
    if (
        len(logits_shape) != 2
        or target_shape != [logits_shape[0]]
        or logits_dtype not in {DType.BFLOAT16, DType.FLOAT32}
        or target_dtype not in {DType.INT32, DType.INT64}
    ):
        return None
    attributes = _attributes(layer)
    reduction = attributes.get("reduction")
    if reduction not in {"mean", "sum"}:
        return None
    weight = attributes.get("weight")
    if weight is not None and weight != "None":
        return None
    label_smoothing = attributes.get("label_smoothing", 0.0)
    if (
        isinstance(label_smoothing, bool)
        or not isinstance(label_smoothing, (int, float))
        or float(label_smoothing) != 0.0
    ):
        return None
    normalized_logits_dtype = (
        TensorDType.BFLOAT16
        if logits_dtype is DType.BFLOAT16
        else TensorDType.FLOAT32
    )
    return CrossEntropyDescriptor(
        rows=logits_shape[0],
        classes=logits_shape[1],
        logits_dtype=normalized_logits_dtype,
        target_dtype=target_dtype,
        reduction=CrossEntropyReduction(str(reduction)),
    )


def _decomposed_cross_entropy_descriptor(
    layers: list[tuple[Mapping[str, object], str]],
) -> CrossEntropyDescriptor | None:
    if [operation for _layer, operation in layers] != [
        "log_softmax",
        "nll_loss_forward",
    ]:
        return None
    log_softmax, nll_loss = layers[0][0], layers[1][0]
    if _softmax_descriptor(
        log_softmax, SoftmaxOperation.LOG_SOFTMAX
    ) is None or not _canonical_nll_loss_arguments(nll_loss):
        return None
    logits = _raw_tensor_metadata(log_softmax, "inputs")
    log_probabilities = _raw_tensor_metadata(log_softmax, "outputs")
    nll_inputs = _raw_tensor_metadata(nll_loss, "inputs")
    nll_outputs = _raw_tensor_metadata(nll_loss, "outputs")
    if (
        len(logits) != 1
        or len(log_probabilities) != 1
        or len(nll_inputs) < 2
        or not nll_outputs
        or nll_inputs[0] != log_probabilities[0]
        or nll_outputs[0][0] != []
    ):
        return None
    logits_shape, logits_dtype = logits[0]
    target_shape, target_dtype = nll_inputs[1]
    if (
        len(logits_shape) != 2
        or target_shape != [logits_shape[0]]
        or logits_dtype not in {DType.BFLOAT16, DType.FLOAT32}
        or target_dtype not in {DType.INT32, DType.INT64}
    ):
        return None
    normalized_logits_dtype = (
        TensorDType.BFLOAT16
        if logits_dtype is DType.BFLOAT16
        else TensorDType.FLOAT32
    )
    return CrossEntropyDescriptor(
        rows=logits_shape[0],
        classes=logits_shape[1],
        logits_dtype=normalized_logits_dtype,
        target_dtype=target_dtype,
        reduction=CrossEntropyReduction.MEAN,
    )


def _canonical_nll_loss_arguments(layer: Mapping[str, object]) -> bool:
    semantic = layer.get("semantic_op")
    if not isinstance(semantic, Mapping):
        return False
    arguments = semantic.get("arguments")
    if not isinstance(arguments, list) or len(arguments) < 5:
        return False
    return (
        _unwrap_semantic_value(arguments[2]) is None
        and _unwrap_semantic_value(arguments[3]) == 1
        and _unwrap_semantic_value(arguments[4]) == -100
    )


def _indexed_read_descriptor(
    layer: Mapping[str, object],
    operation: IndexedReadOperation,
) -> IndexedReadDescriptor | None:
    inputs = _raw_tensor_metadata(layer, "inputs")
    if len(inputs) < 2:
        return None
    source, index = inputs[0], inputs[1]
    if operation is IndexedReadOperation.EMBEDDING:
        source, index = inputs[0], inputs[1]
        axis = 0
    else:
        axis = _normalized_axis(
            _attributes(layer).get("dim"),
            len(source[0]),
        )
    if (
        axis is None
        or source[1] not in {DType.FLOAT16, DType.BFLOAT16, DType.FLOAT32}
        or index[1] not in {DType.INT32, DType.INT64}
    ):
        return None
    return IndexedReadDescriptor(
        operation=operation,
        source_shape=source[0],
        index_shape=index[0],
        axis=axis,
        payload_dtype=TensorDType(source[1].value),
        index_dtype=index[1],
        element_bytes=_dtype_bytes(TensorDType(source[1].value)),
    )


def _indexed_update_descriptor(
    layer: Mapping[str, object],
    operation: IndexedUpdateOperation,
) -> IndexedUpdateDescriptor | None:
    inputs = _raw_tensor_metadata(layer, "inputs")
    if len(inputs) < 3:
        return None
    output, index = inputs[0], inputs[1]
    axis = _normalized_axis(_attributes(layer).get("dim"), len(output[0]))
    if (
        axis is None
        or output[1] is not DType.FLOAT32
        or index[1] not in {DType.INT32, DType.INT64}
    ):
        return None
    return IndexedUpdateDescriptor(
        operation=operation,
        output_shape=output[0],
        index_shape=index[0],
        axis=axis,
        payload_dtype=TensorDType.FLOAT32,
        index_dtype=index[1],
        atomic=operation
        in {
            IndexedUpdateOperation.SCATTER_ADD,
            IndexedUpdateOperation.INDEX_ADD,
        },
    )


def _composite_descriptor(
    layers: Mapping[str, object],
    operation_layers: list[tuple[Mapping[str, object], str]],
    metadata: Mapping[str, object],
) -> CompositeGraphDescriptor | None:
    if not 2 <= len(operation_layers) <= _MAX_COMPOSITE_NODES:
        return None
    operation_names = [
        name
        for name, layer in layers.items()
        if any(layer is item for item, _operation in operation_layers)
    ]
    if len(operation_names) != len(operation_layers):
        return None
    nodes: list[CompositeGraphNode] = []
    for name, (layer, operation) in zip(
        operation_names,
        operation_layers,
        strict=True,
    ):
        descriptor = _primitive_layer_descriptor(layer, operation)
        inputs = _tensor_names(layer, "inputs")
        outputs = _tensor_names(layer, "outputs")
        if descriptor is None or not outputs:
            return None
        nodes.append(
            CompositeGraphNode(
                node_id=name,
                layer_names=[name],
                descriptor=descriptor,
                input_tensors=inputs,
                output_tensors=outputs,
            )
        )
    regions = _fusion_regions(metadata.get("fusion"))
    edges = _composite_edges(layers, operation_names, regions)
    if edges is None:
        return None
    schedule = _topological_schedule(operation_names, edges)
    if schedule is None:
        return None
    return CompositeGraphDescriptor(
        graph_class=_composite_graph_class(metadata, nodes),
        nodes=nodes,
        edges=edges,
        schedule=schedule,
    )


def _primitive_layer_descriptor(
    layer: Mapping[str, object],
    operation: str,
) -> object | None:
    if operation in _ELEMENTWISE_TYPES:
        return _elementwise_descriptor([(layer, operation)])
    if operation in _TRANSPOSE_TYPES:
        return _transpose_descriptor(layer)
    if operation in _REDUCTION_TYPES:
        return _reduction_descriptor(layer, _REDUCTION_TYPES[operation])
    if operation in _MATMUL_TYPES:
        return _matmul_descriptor(layer, batched=operation == "bmm")
    if operation in _SOFTMAX_TYPES:
        return _softmax_descriptor(layer, _SOFTMAX_TYPES[operation])
    if operation == "cross_entropy":
        return _cross_entropy_descriptor(layer)
    if operation in _INDEXED_READ_TYPES:
        return _indexed_read_descriptor(layer, _INDEXED_READ_TYPES[operation])
    if operation in _INDEXED_UPDATE_TYPES:
        return _indexed_update_descriptor(
            layer,
            _INDEXED_UPDATE_TYPES[operation],
        )
    return None


def _composite_edges(
    layers: Mapping[str, object],
    operation_names: list[str],
    regions: list[FusionRegion],
) -> list[CompositeGraphEdge] | None:
    admitted = set(operation_names)
    region_by_layer = {
        layer_name: region.region_id
        for region in regions
        for layer_name in region.layer_names
    }
    edges: list[CompositeGraphEdge] = []
    for consumer in operation_names:
        predecessors = _admitted_predecessors(
            layers,
            consumer,
            admitted,
        )
        if predecessors is None:
            return None
        for producer in predecessors:
            producer_layer = layers[producer]
            if not isinstance(producer_layer, Mapping):
                return None
            tensors = _tensor_names(
                cast("Mapping[str, object]", producer_layer),
                "outputs",
            )
            if len(tensors) != 1:
                return None
            edges.append(
                CompositeGraphEdge(
                    producer=producer,
                    consumer=consumer,
                    tensor=tensors[0],
                    materialized=(
                        region_by_layer.get(producer)
                        != region_by_layer.get(consumer)
                    ),
                )
            )
    identities = {(edge.producer, edge.consumer, edge.tensor) for edge in edges}
    return edges if len(identities) == len(edges) else None


def _admitted_predecessors(
    layers: Mapping[str, object],
    consumer: str,
    admitted: set[str],
) -> list[str] | None:
    """Resolve operation predecessors through transparent IR-only layers."""
    pending = [consumer]
    visited = {consumer}
    resolved: list[str] = []
    while pending:
        current = pending.pop()
        layer = layers.get(current)
        if not isinstance(layer, Mapping):
            return None
        connections = layer.get("connections")
        if not isinstance(connections, Mapping):
            return None
        raw = connections.get("inputs")
        if not isinstance(raw, list):
            return None
        for predecessor in (str(item) for item in raw):
            if predecessor in admitted:
                resolved.append(predecessor)
            elif predecessor in layers and predecessor not in visited:
                visited.add(predecessor)
                pending.append(predecessor)
    return list(dict.fromkeys(resolved))


def _topological_schedule(
    node_ids: list[str],
    edges: list[CompositeGraphEdge],
) -> list[str] | None:
    incoming = dict.fromkeys(node_ids, 0)
    successors = {node_id: [] for node_id in node_ids}
    for edge in edges:
        incoming[edge.consumer] += 1
        successors[edge.producer].append(edge.consumer)
    roots = sorted(node_id for node_id, count in incoming.items() if count == 0)
    if len(roots) != 1:
        return None
    ready = roots
    schedule: list[str] = []
    while ready:
        node_id = ready.pop(0)
        schedule.append(node_id)
        for successor in sorted(successors[node_id]):
            incoming[successor] -= 1
            if incoming[successor] == 0:
                ready.append(successor)
        ready.sort()
    return schedule if len(schedule) == len(node_ids) else None


def _is_minigpt_semantics(
    metadata: Mapping[str, object],
    nodes: list[CompositeGraphNode],
) -> bool:
    semantics = metadata.get("performance_semantics")
    if not isinstance(semantics, Mapping):
        return False
    values = cast("Mapping[str, object]", semantics)
    sequence_length = values.get("sequence_length")
    descriptors = [node.descriptor for node in nodes]
    return (
        values.get("graph_class") == "transformer_block"
        and values.get("hidden_size") == 768
        and values.get("num_heads") == 8
        and isinstance(sequence_length, int)
        and not isinstance(sequence_length, bool)
        and 0 < sequence_length <= 1024
        and values.get("dtype") in {"float32", "torch.float32"}
        and any(
            isinstance(descriptor, MatmulDescriptor)
            and descriptor.input_dtype is TensorDType.FLOAT32
            and 768 in {descriptor.m, descriptor.n, descriptor.k}
            for descriptor in descriptors
        )
        and any(
            isinstance(descriptor, SoftmaxDescriptor)
            and descriptor.reduction_width <= 1024
            for descriptor in descriptors
        )
        and any(
            isinstance(descriptor, ReductionDescriptor)
            and descriptor.operation is ReductionOperation.LAYER_NORM
            and descriptor.reduction_width == 768
            for descriptor in descriptors
        )
    )


def _composite_graph_class(
    metadata: Mapping[str, object],
    nodes: list[CompositeGraphNode],
) -> Literal["composite_graph", "transformer_block", "concurrent_graph"]:
    if _is_minigpt_semantics(metadata, nodes):
        return "transformer_block"
    semantics = metadata.get("performance_semantics")
    if (
        isinstance(semantics, Mapping)
        and semantics.get("graph_class") == "concurrent_graph"
    ):
        return "concurrent_graph"
    return "composite_graph"


def _attributes(layer: Mapping[str, object]) -> Mapping[str, object]:
    value = layer.get("attributes")
    if isinstance(value, Mapping):
        return cast("Mapping[str, object]", value)
    semantic = layer.get("semantic_op")
    if not isinstance(semantic, Mapping):
        return {}
    kwargs = semantic.get("kwargs", semantic.get("attributes"))
    result = (
        {
            str(name): _unwrap_semantic_value(item)
            for name, item in kwargs.items()
        }
        if isinstance(kwargs, Mapping)
        else {}
    )
    arguments = semantic.get("arguments", semantic.get("operands"))
    positional = [
        _unwrap_semantic_value(item)
        for item in (arguments if isinstance(arguments, list) else [])
        if not (isinstance(item, Mapping) and "tensor" in item)
    ]
    operation = str(layer.get("type", "")).lower()
    if operation in {
        *_SOFTMAX_TYPES,
        *_INDEXED_READ_TYPES,
        *_INDEXED_UPDATE_TYPES,
    }:
        if "dim" not in result and positional:
            result["dim"] = positional[0]
    elif operation == "layer_norm" and positional:
        result.setdefault("normalized_shape", positional[0])
    return result


def _unwrap_semantic_value(value: object) -> object:
    if isinstance(value, Mapping) and "value" in value:
        mapping = cast("Mapping[str, object]", value)
        return _unwrap_semantic_value(mapping["value"])
    if isinstance(value, list):
        return [_unwrap_semantic_value(item) for item in value]
    return value


def _normalized_axis(value: object, rank: int) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    normalized = value + rank if value < 0 else value
    return normalized if 0 <= normalized < rank else None


def _tensor_names(
    layer: Mapping[str, object],
    direction: str,
) -> list[str]:
    names = layer.get("tensor_names")
    if not isinstance(names, Mapping):
        return []
    values = names.get(direction)
    return (
        [str(value) for value in values]
        if isinstance(values, list)
        and all(isinstance(value, str) for value in values)
        else []
    )


def _raw_tensor_metadata(
    layer: Mapping[str, object],
    direction: str,
) -> list[tuple[list[int], DType]]:
    shapes = layer.get("tensor_shapes")
    dtypes = layer.get("tensor_dtypes")
    if not isinstance(shapes, Mapping) or not isinstance(dtypes, Mapping):
        return []
    raw_shapes = shapes.get(direction)
    raw_dtypes = dtypes.get(direction)
    if not isinstance(raw_shapes, list) or not isinstance(raw_dtypes, list):
        return []
    if len(raw_shapes) != len(raw_dtypes):
        return []
    result: list[tuple[list[int], DType]] = []
    for shape, dtype in zip(raw_shapes, raw_dtypes, strict=True):
        if not (
            isinstance(shape, list)
            and all(
                isinstance(dimension, int) and dimension > 0
                for dimension in shape
            )
        ):
            return []
        try:
            result.append(
                (
                    cast("list[int]", shape),
                    DType(str(dtype).removeprefix("torch.")),
                )
            )
        except ValueError:
            return []
    return result


def _tensor_metadata(
    layer: Mapping[str, object],
    direction: str,
) -> list[tuple[list[int], TensorDType]]:
    shapes = layer.get("tensor_shapes")
    dtypes = layer.get("tensor_dtypes")
    if not isinstance(shapes, Mapping) or not isinstance(dtypes, Mapping):
        return []
    raw_shapes = shapes.get(direction)
    raw_dtypes = dtypes.get(direction)
    if not isinstance(raw_shapes, list) or not isinstance(raw_dtypes, list):
        return []
    if len(raw_shapes) != len(raw_dtypes):
        return []
    result: list[tuple[list[int], TensorDType]] = []
    for shape, dtype in zip(raw_shapes, raw_dtypes, strict=True):
        if (
            isinstance(shape, list)
            and shape
            and all(
                isinstance(dimension, int) and dimension > 0
                for dimension in shape
            )
        ):
            try:
                normalized_dtype = str(dtype).removeprefix("torch.")
                result.append(
                    (
                        cast("list[int]", shape),
                        TensorDType(normalized_dtype),
                    )
                )
            except ValueError:
                return []
    return result


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"solar analysis requires mapping {name!r}")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"solar analysis mapping {name!r} has non-string key")
    return {str(key): item for key, item in value.items()}


def _nonnegative_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"solar analysis requires numeric {name!r}")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"solar analysis requires finite nonnegative {name!r}")
    return number


def _resource_work(value: object) -> dict[str, dict[str, float]]:
    resources = _mapping(value, "total.resource_work")
    return {
        resource: {
            mode: _nonnegative_number(
                amount, f"resource_work.{resource}.{mode}"
            )
            for mode, amount in _mapping(
                raw_modes,
                f"resource_work.{resource}",
            ).items()
        }
        for resource, raw_modes in resources.items()
    }


def _semantic_bytes(total: Mapping[str, object]) -> float:
    for key in ("prefetched_bytes", "fused_bytes", "model_io_bytes"):
        if (value := total.get(key)) is not None:
            return _nonnegative_number(value, f"total.{key}")
    raise ValueError("solar analysis lacks semantic byte count")


def _fusion_regions(raw_fusion: object) -> list[FusionRegion]:
    if not isinstance(raw_fusion, Mapping):
        return []
    raw_regions = raw_fusion.get("regions")
    if not isinstance(raw_regions, list):
        return []
    return [
        _fusion_region(cast("Mapping[object, object]", region), index)
        for index, region in enumerate(raw_regions)
        if isinstance(region, Mapping)
    ]


def _fusion_region(region: Mapping[object, object], index: int) -> FusionRegion:
    raw_layers = region.get("layers")
    return FusionRegion(
        region_id=str(region.get("id") or f"semantic_region_{index}"),
        layer_names=(
            [str(layer) for layer in raw_layers]
            if isinstance(raw_layers, list)
            else []
        ),
    )


def _fusion_regions_cover_operations(
    layers: Mapping[str, object],
    regions: list[FusionRegion],
) -> bool:
    operation_names = {
        name
        for name, layer in layers.items()
        if isinstance(layer, Mapping)
        and str(layer.get("type", "")).lower() not in _IGNORED_TYPES
    }
    cited = [
        layer_name
        for region in regions
        for layer_name in region.layer_names
        if layer_name in operation_names
    ]
    return set(cited) == operation_names and len(cited) == len(set(cited))


def _shape_elements(shape: list[int]) -> int:
    return math.prod(shape)


def _dtype_bytes(dtype: TensorDType) -> int:
    return 4 if dtype is TensorDType.FLOAT32 else 2


__all__ = [
    "SOLAR_ANALYSIS_SCHEMA_VERSION",
    "SOLAR_REQUEST_MANIFEST_SCHEMA_VERSION",
    "load_manifest_semantic_characterization",
    "load_semantic_characterization",
]

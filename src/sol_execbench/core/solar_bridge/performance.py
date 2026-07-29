# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Validated SOLAR-to-performance-model boundary."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import yaml

from sol_execbench.core.bench.performance_model.models import (
    ElementwiseDescriptor,
    ElementwiseOperationClass,
    EvidenceReference,
    FusionRegion,
    MatmulDescriptor,
    ReductionDescriptor,
    ReductionOperation,
    SemanticCharacterization,
    TensorDType,
    TransposeDescriptor,
    UnsupportedDescriptor,
    WorkloadKind,
)
from sol_execbench.core.integrity import (
    sha256_file,
    validate_relative_artifact_path,
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
}
_MATMUL_TYPES = frozenset({"bmm", "gemm", "matmul", "mm"})
_IGNORED_TYPES = frozenset({"", "start", "input"})


def load_manifest_semantic_characterization(
    manifest_path: str | Path,
    *,
    workload_uuid: str,
    definition: str,
) -> SemanticCharacterization:
    """Verify a SOLAR manifest and load its cited analysis."""
    path = Path(manifest_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    manifest = _mapping(raw, "manifest")
    if manifest.get("schema_version") != SOLAR_REQUEST_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported_solar_manifest_schema")
    if manifest.get("analysis_id") != f"{definition}:{workload_uuid}":
        raise ValueError("solar_manifest_workload_identity_mismatch")
    if manifest.get("sol_score_eligible") is not True:
        raise ValueError("solar_manifest_bound_not_eligible")
    analysis_path, digest = _manifest_analysis(path.parent, manifest)
    return load_semantic_characterization(
        analysis_path,
        workload_uuid=workload_uuid,
        expected_sha256=digest,
    )


def _manifest_analysis(
    root: Path,
    manifest: Mapping[str, object],
) -> tuple[Path, str]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("solar_manifest_artifacts_invalid")
    matches = [
        item
        for item in artifacts
        if isinstance(item, Mapping)
        and item.get("path") == "solar-analysis.yaml"
    ]
    if len(matches) != 1:
        raise ValueError("solar_manifest_analysis_reference_missing")
    record = matches[0]
    relative = validate_relative_artifact_path(record.get("path"))
    digest = str(record.get("sha256"))
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
    descriptor, kind, reasons = _semantic_descriptor(layers)
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
) -> tuple[object, WorkloadKind, list[str]]:
    operation_layers = _operation_layers(layers)
    operation_types = [item[1] for item in operation_layers]
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
    reasons = ["unsupported_workload_descriptor"]
    return (
        UnsupportedDescriptor(reason_codes=reasons),
        WorkloadKind.UNSUPPORTED,
        reasons,
    )


def _operation_layers(
    layers: Mapping[str, object],
) -> list[tuple[Mapping[str, object], str]]:
    return [
        (cast("Mapping[str, object]", layer), operation)
        for layer in layers.values()
        if isinstance(layer, Mapping)
        and (operation := str(layer.get("type", "")).lower())
        not in _IGNORED_TYPES
    ]


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
        if operation is ReductionOperation.RMS_NORM
        else {
            tuple(input_shape[:-1]),
            (*input_shape[:-1], 1),
        }
    )
    if tuple(output_shape) not in expected_outputs:
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
    if (
        left[1] is not TensorDType.FLOAT16
        or right[1] is not TensorDType.FLOAT16
        or output_dtype is not TensorDType.FLOAT32
    ):
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
    )


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

# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Validated SOLAR-to-performance-model boundary.

Only this bridge imports the inner ``solar`` package. Performance-model callers
consume the frozen outer characterization instead of depending on SOLAR internals.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

from sol_execbench.core.bench.performance_model.models import (
    EvidenceReference,
    FusionRegion,
    SemanticCharacterization,
    WorkloadKind,
)
from sol_execbench.core.integrity import sha256_file
from solar.artifacts import load_yaml_artifact
from solar.schema_versions import SOLAR_ANALYSIS_SCHEMA_VERSION

_ELEMENTWISE_TYPES = frozenset(
    {
        "abs",
        "add",
        "clamp",
        "cos",
        "div",
        "exp",
        "gelu",
        "log",
        "mul",
        "neg",
        "relu",
        "sigmoid",
        "sin",
        "sub",
        "tanh",
    },
)
_TRANSPOSE_TYPES = frozenset({"permute", "transpose"})
_REDUCTION_TYPES = frozenset(
    {"layer_norm", "mean", "norm", "reduce", "rms_norm", "sum"},
)
_MATMUL_TYPES = frozenset(
    {"bmm", "einsum", "gemm", "linear", "matmul", "mm"},
)


def load_semantic_characterization(
    path: str | Path,
    *,
    workload_uuid: str,
    expected_sha256: str | None = None,
) -> SemanticCharacterization:
    """Load and strictly characterize one canonical ``solar-analysis.yaml``."""
    source_path = Path(path)
    actual_sha256 = sha256_file(source_path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError("solar_analysis_sha256_mismatch")
    document = load_yaml_artifact(source_path)
    data = document.data
    if data.get("schema_version") != SOLAR_ANALYSIS_SCHEMA_VERSION:
        raise ValueError("unsupported_solar_analysis_schema")
    layers = _mapping(data.get("layers"), "layers")
    total = _mapping(data.get("total"), "total")
    metadata = _mapping(data.get("metadata"), "metadata")
    resource_work = _resource_work(total.get("resource_work"))
    workload_kind = _workload_kind(layers)
    reasons = (
        ["unsupported_workload_kind"]
        if workload_kind is WorkloadKind.UNSUPPORTED
        else []
    )
    return SemanticCharacterization(
        workload_uuid=workload_uuid,
        workload_kind=workload_kind,
        shape=_representative_shape(layers),
        resource_work=resource_work,
        fusion_regions=_fusion_regions(metadata.get("fusion"), layers),
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


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"solar analysis requires mapping {name!r}")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(
                f"solar analysis mapping {name!r} has non-string key"
            )
        result[key] = item
    return result


def _nonnegative_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"solar analysis requires numeric {name!r}")
    number = float(value)
    if number < 0:
        raise ValueError(f"solar analysis requires nonnegative {name!r}")
    return number


def _resource_work(value: object) -> dict[str, dict[str, float]]:
    resources = _mapping(value, "total.resource_work")
    result: dict[str, dict[str, float]] = {}
    for resource, raw_modes in resources.items():
        modes = _mapping(raw_modes, f"resource_work.{resource}")
        result[resource] = {
            mode: _nonnegative_number(
                amount, f"resource_work.{resource}.{mode}"
            )
            for mode, amount in modes.items()
        }
    return result


def _semantic_bytes(total: Mapping[str, object]) -> float:
    for key in ("prefetched_bytes", "fused_bytes", "model_io_bytes"):
        value = total.get(key)
        if value is not None:
            return _nonnegative_number(value, f"total.{key}")
    raise ValueError("solar analysis lacks semantic byte count")


def _workload_kind(layers: Mapping[str, object]) -> WorkloadKind:
    operation_types = {
        str(layer.get("type", "")).lower()
        for layer in layers.values()
        if isinstance(layer, Mapping)
        and str(layer.get("type", "")).lower() not in {"", "start", "input"}
    }
    if operation_types & _MATMUL_TYPES:
        return WorkloadKind.MATMUL
    if operation_types & _REDUCTION_TYPES:
        return WorkloadKind.REDUCTION
    if operation_types & _TRANSPOSE_TYPES:
        return WorkloadKind.TRANSPOSE
    if operation_types and operation_types <= _ELEMENTWISE_TYPES:
        return WorkloadKind.ELEMENTWISE
    return WorkloadKind.UNSUPPORTED


def _representative_shape(layers: Mapping[str, object]) -> list[int]:
    candidates: list[list[int]] = []
    for layer in layers.values():
        if not isinstance(layer, Mapping):
            continue
        shapes = layer.get("tensor_shapes")
        if not isinstance(shapes, Mapping):
            continue
        for direction in ("outputs", "inputs"):
            values = shapes.get(direction)
            if not isinstance(values, list):
                continue
            for shape in values:
                if isinstance(shape, list) and all(
                    isinstance(dimension, int) and dimension >= 0
                    for dimension in shape
                ):
                    candidates.append(cast("list[int]", shape))
    return max(candidates, key=lambda shape: _shape_elements(shape), default=[])


def _shape_elements(shape: list[int]) -> int:
    elements = 1
    for dimension in shape:
        elements *= dimension
    return elements


def _fusion_regions(
    raw_fusion: object,
    layers: Mapping[str, object],
) -> list[FusionRegion]:
    if isinstance(raw_fusion, Mapping):
        raw_regions = raw_fusion.get("regions")
        if isinstance(raw_regions, list):
            regions = [
                _fusion_region(
                    cast("Mapping[object, object]", region),
                    index,
                )
                for index, region in enumerate(raw_regions)
                if isinstance(region, Mapping)
            ]
            if regions:
                return regions
    layer_names = [
        name
        for name, layer in layers.items()
        if isinstance(layer, Mapping)
        and str(layer.get("type", "")).lower() not in {"start", "input"}
    ]
    return [
        FusionRegion(region_id="semantic_region_0", layer_names=layer_names)
    ]


def _fusion_region(region: Mapping[object, object], index: int) -> FusionRegion:
    region_id = str(region.get("id") or f"semantic_region_{index}")
    raw_layers = region.get("layers")
    layer_names = (
        [str(layer) for layer in raw_layers]
        if isinstance(raw_layers, list)
        else []
    )
    return FusionRegion(region_id=region_id, layer_names=layer_names)


__all__ = [
    "SOLAR_ANALYSIS_SCHEMA_VERSION",
    "load_semantic_characterization",
]

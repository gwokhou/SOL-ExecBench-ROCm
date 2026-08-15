# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Repository-owned comparison of fixed SOLAR IR-path accounting."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from sol_execbench.core.data.json_utils import atomic_write_json_value
from sol_execbench.core.integrity import sha256_file
from sol_execbench.core.integrity.artifacts import (
    validate_relative_artifact_path,
    validate_sha256,
)
from sol_execbench.core.solar_bridge.models import (
    IRPath,
    formal_artifact_paths,
)
from sol_execbench.core.solar_bridge.path_comparison_models import (
    ComparisonSection,
    CrossPathComparisonResult,
    DifferenceCategory,
    WorkloadPathComparison,
)
from solar.artifacts import load_yaml_artifact
from solar.schema_versions import (
    SOLAR_ANALYSIS_SCHEMA_VERSION,
    SOLAR_REQUEST_MANIFEST_SCHEMA_VERSION,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class _WorkloadArtifacts:
    relative: Path
    directory: Path
    manifest: dict[str, object]
    attestation: dict[str, object]
    analysis: dict[str, object]
    graph: dict[str, object]


def compare_solar_ir_paths(
    make_fx_root: Path,
    torchview_root: Path,
    output: Path,
) -> CrossPathComparisonResult:
    """Compare two analyzed roots without selecting a favorable result."""
    expected = (
        (IRPath.MAKE_FX_ATEN, make_fx_root.resolve()),
        (IRPath.TORCHVIEW_EXTENDED_EINSUM, torchview_root.resolve()),
    )
    loaded = {ir_path: _load_root(root, ir_path) for ir_path, root in expected}
    left_path, right_path = (ir_path for ir_path, _ in expected)
    path_names: tuple[str, str] = (left_path.value, right_path.value)
    left = loaded[left_path]
    right = loaded[right_path]
    common = sorted(left.keys() & right.keys())
    comparisons = tuple(
        _compare_workload(
            relative,
            left[relative],
            right[relative],
            path_names,
        )
        for relative in common
    )
    result = CrossPathComparisonResult(
        roots={
            ir_path.value: {
                "root": str(root),
                "available_workloads": len(loaded[ir_path]),
            }
            for ir_path, root in expected
        },
        missing_by_path={
            left_path.value: tuple(
                str(item) for item in sorted(right.keys() - left.keys())
            ),
            right_path.value: tuple(
                str(item) for item in sorted(left.keys() - right.keys())
            ),
        },
        comparisons=comparisons,
    )
    atomic_write_json_value(output, result.to_dict())
    return result


def _load_root(
    root: Path,
    expected_path: IRPath,
) -> dict[Path, _WorkloadArtifacts]:
    if not root.is_dir():
        raise ValueError(f"analysis root is not a directory: {root}")
    workloads: dict[Path, _WorkloadArtifacts] = {}
    for manifest_path in sorted(root.rglob("manifest.yaml")):
        relative = manifest_path.parent.relative_to(root)
        if relative in workloads:
            raise ValueError(f"duplicate workload directory: {relative}")
        workloads[relative] = _load_workload(
            relative,
            manifest_path.parent,
            expected_path,
        )
    if not workloads:
        raise ValueError(
            f"analysis root contains no workload manifests: {root}"
        )
    return workloads


def _load_workload(
    relative: Path,
    directory: Path,
    expected_path: IRPath,
) -> _WorkloadArtifacts:
    manifest = _load_mapping(directory / "manifest.yaml")
    if manifest.get("schema_version") != SOLAR_REQUEST_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"{relative}: unsupported request manifest schema")
    contract = _mapping(manifest.get("analysis_contract"), "analysis_contract")
    observed_path = IRPath(str(contract.get("ir_path", "")))
    if observed_path is not expected_path:
        raise ValueError(
            f"{relative}: expected {expected_path.value}, "
            f"found {observed_path.value}",
        )
    artifacts = _verify_manifest_artifacts(directory, manifest, expected_path)
    analysis = _load_mapping(artifacts["solar-analysis.yaml"])
    if analysis.get("schema_version") != SOLAR_ANALYSIS_SCHEMA_VERSION:
        raise ValueError(f"{relative}: unsupported SOLAR analysis schema")
    attestation = _load_mapping(
        artifacts["conversion-attestation.yaml"],
    )
    _validate_attestation_identity(
        manifest,
        attestation,
        expected_path,
    )
    return _WorkloadArtifacts(
        relative=relative,
        directory=directory,
        manifest=manifest,
        attestation=attestation,
        analysis=analysis,
        graph=_load_mapping(artifacts[expected_path.graph_filename]),
    )


def _verify_manifest_artifacts(
    directory: Path,
    manifest: Mapping[str, object],
    ir_path: IRPath,
) -> dict[str, Path]:
    records = _list(manifest.get("artifacts"), "manifest artifacts")
    paths: dict[str, Path] = {}
    for raw in records:
        record = _mapping(raw, "artifact")
        relative = validate_relative_artifact_path(record.get("path"))
        digest = validate_sha256(record.get("sha256"))
        path = directory / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"analysis artifact is missing: {relative}")
        if sha256_file(path) != digest:
            raise ValueError(f"analysis artifact SHA-256 mismatch: {relative}")
        paths[relative] = path
    required = formal_artifact_paths(ir_path)
    if not required.issubset(paths):
        raise ValueError(
            "analysis artifact set is incomplete: "
            f"required {sorted(required)}, found {sorted(paths)}",
        )
    return paths


def _validate_attestation_identity(
    manifest: Mapping[str, object],
    attestation: Mapping[str, object],
    ir_path: IRPath,
) -> None:
    records = _list(manifest.get("artifacts"), "manifest artifacts")
    artifact_digests = {
        str(record.get("path")): str(record.get("sha256"))
        for raw in records
        if isinstance(raw, Mapping)
        for record in (raw,)
    }
    reference = _mapping(manifest.get("reference"), "reference")
    expected = {
        str(reference.get("name")): str(reference.get("sha256")),
        ir_path.graph_filename: artifact_digests[ir_path.graph_filename],
    }
    subjects = _list(attestation.get("subject"), "attestation subject")
    observed = {
        str(subject.get("name")): str(
            _mapping(subject.get("digest"), "subject digest").get("sha256"),
        )
        for raw in subjects
        for subject in (_mapping(raw, "attestation subject"),)
    }
    if any(observed.get(name) != digest for name, digest in expected.items()):
        raise ValueError("conversion attestation subject identity mismatch")
    predicate = _mapping(attestation.get("predicate"), "predicate")
    if predicate.get("status") != "passed":
        raise ValueError("conversion attestation did not pass")


def _load_mapping(path: Path) -> dict[str, object]:
    return dict(load_yaml_artifact(path).data)


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise ValueError(f"{field} must be a mapping")
    return cast("Mapping[str, object]", value)


def _list(value: object, field: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    return list(value)


def _compare_workload(
    relative: Path,
    left: _WorkloadArtifacts,
    right: _WorkloadArtifacts,
    path_names: tuple[str, str],
) -> WorkloadPathComparison:
    external = _section(
        _external_reference_io(left),
        _external_reference_io(right),
        path_names,
        DifferenceCategory.EXTRACTION_TOPOLOGY_LOSS,
    )
    model_io = _model_io_section(left, right, path_names)
    resources = _resource_section(left, right, path_names)
    fusion = _section(
        _fusion_intermediates(left),
        _fusion_intermediates(right),
        path_names,
        DifferenceCategory.DIALECT_DECOMPOSITION_DIFFERENCE,
    )
    bound_category = _bound_category(external, model_io, resources, fusion)
    bound = _section(
        _formal_bound(left),
        _formal_bound(right),
        path_names,
        bound_category,
    )
    return WorkloadPathComparison(
        workload=str(relative),
        external_reference_io=external,
        model_io_accounting=model_io,
        mandatory_resource_work=resources,
        fusion_intermediate_accounting=fusion,
        formal_bound=bound,
    )


def _section(
    left: dict[str, object],
    right: dict[str, object],
    path_names: tuple[str, str],
    category: DifferenceCategory,
) -> ComparisonSection:
    differences = tuple(
        key
        for key in sorted(left.keys() | right.keys())
        if left.get(key) != right.get(key)
    )
    return ComparisonSection(
        match=not differences,
        classification=category if differences else None,
        differences=differences,
        values={path_names[0]: left, path_names[1]: right},
    )


def _external_reference_io(item: _WorkloadArtifacts) -> dict[str, object]:
    manifest = item.manifest
    contract = _mapping(manifest.get("analysis_contract"), "analysis_contract")
    predicate = _mapping(
        item.attestation.get("predicate"),
        "attestation predicate",
    )
    return {
        "analysis_id": manifest.get("analysis_id"),
        "architecture_sha256": manifest.get("architecture_sha256"),
        "reference": manifest.get("reference"),
        "precision": contract.get("precision"),
        "trace_seed": contract.get("trace_seed"),
        "verification": {
            key: contract.get(key)
            for key in (
                "verification_seeds",
                "atol",
                "rtol",
                "required_matched_ratio",
                "max_error_cap",
                "allow_negative_inf",
                "preserved_input_indices",
            )
        },
        "attestation_status": predicate.get("status"),
        "attestation_verifier": predicate.get("verifier"),
        "verification_execution": predicate.get("execution"),
        "verification_cases": predicate.get("cases"),
        "input_signature": _input_signature(item.graph),
        "output_signature": _output_signature(item.graph),
    }


def _input_signature(
    graph: Mapping[str, object],
) -> list[dict[str, object]]:
    inputs: dict[int, dict[str, object]] = {}
    layers = _mapping(graph.get("layers"), "graph layers")
    live_layer_ids = _live_layer_ids(graph)
    for layer_id, raw_layer in layers.items():
        layer = _mapping(raw_layer, "graph layer")
        if (
            layer_id not in live_layer_ids
            or str(layer.get("type", "")).lower() != "start"
        ):
            continue
        index = layer.get("source_input_index")
        if not isinstance(index, int) or index < 0 or index in inputs:
            raise ValueError(
                "graph inputs lack unique source_input_index values"
            )
        shapes = _mapping(layer.get("tensor_shapes"), "tensor_shapes")
        dtypes = _mapping(layer.get("tensor_dtypes"), "tensor_dtypes")
        outputs = _list(shapes.get("outputs"), "input shapes")
        output_dtypes = _list(dtypes.get("outputs"), "input dtypes")
        if len(outputs) != 1 or len(output_dtypes) != 1:
            raise ValueError("graph input must have one exact tensor signature")
        inputs[index] = {
            "source_input_index": index,
            "shape": outputs[0],
            "dtype": output_dtypes[0],
        }
    return [inputs[index] for index in sorted(inputs)]


def _live_layer_ids(graph: Mapping[str, object]) -> set[str]:
    layers = _mapping(graph.get("layers"), "graph layers")
    producers: dict[str, str] = {}
    for layer_id, raw_layer in layers.items():
        layer = _mapping(raw_layer, "graph layer")
        names = _mapping(layer.get("tensor_names"), "tensor_names")
        for name in _list(names.get("outputs"), "tensor output names"):
            producers[str(name)] = str(layer_id)
    declared = _list(graph.get("outputs"), "graph outputs")
    if not declared:
        raise ValueError("graph has no ordered declared outputs")
    pending = [
        producers[str(name)] for name in declared if str(name) in producers
    ]
    if len(pending) != len(declared):
        raise ValueError("declared graph output has no exact producer")
    live: set[str] = set()
    while pending:
        layer_id = pending.pop()
        if layer_id in live:
            continue
        live.add(layer_id)
        layer = _mapping(layers[layer_id], "graph layer")
        connections = _mapping(layer.get("connections") or {}, "connections")
        pending.extend(
            str(item)
            for item in _list(connections.get("inputs"), "connections.inputs")
            if str(item) in layers
        )
    return live


def _output_signature(
    graph: Mapping[str, object],
) -> list[dict[str, object]]:
    declared = _list(graph.get("outputs"), "graph outputs")
    if not declared:
        raise ValueError("graph has no ordered declared outputs")
    producers: dict[str, tuple[object, object]] = {}
    layers = _mapping(graph.get("layers"), "graph layers")
    for raw_layer in layers.values():
        layer = _mapping(raw_layer, "graph layer")
        names = _mapping(layer.get("tensor_names"), "tensor_names")
        shapes = _mapping(layer.get("tensor_shapes"), "tensor_shapes")
        dtypes = _mapping(layer.get("tensor_dtypes"), "tensor_dtypes")
        for name, shape, dtype in zip(
            _list(names.get("outputs"), "tensor output names"),
            _list(shapes.get("outputs"), "tensor output shapes"),
            _list(dtypes.get("outputs"), "tensor output dtypes"),
            strict=True,
        ):
            if str(name) in producers:
                raise ValueError(f"duplicate tensor producer: {name}")
            producers[str(name)] = (shape, dtype)
    signatures: list[dict[str, object]] = []
    for slot, name in enumerate(declared):
        if str(name) not in producers:
            raise ValueError(f"declared output has no producer: {name}")
        shape, dtype = producers[str(name)]
        signatures.append({"slot": slot, "shape": shape, "dtype": dtype})
    return signatures


def _total(item: _WorkloadArtifacts) -> Mapping[str, object]:
    return _mapping(item.analysis.get("total"), "analysis total")


def _metadata(item: _WorkloadArtifacts) -> Mapping[str, object]:
    return _mapping(item.analysis.get("metadata"), "analysis metadata")


def _model_io(item: _WorkloadArtifacts) -> dict[str, object]:
    total = _total(item)
    return {
        key: total.get(key)
        for key in (
            "model_io_elements",
            "model_io_bytes",
            "fused_elements",
            "fused_bytes",
        )
    }


def _model_io_section(
    left: _WorkloadArtifacts,
    right: _WorkloadArtifacts,
    path_names: tuple[str, str],
) -> ComparisonSection:
    left_value = _model_io(left)
    right_value = _model_io(right)
    fused_matches = all(
        left_value[key] == right_value[key]
        for key in ("fused_elements", "fused_bytes")
    )
    category = (
        DifferenceCategory.DIALECT_DECOMPOSITION_DIFFERENCE
        if fused_matches
        else DifferenceCategory.RESOURCE_MODEL_BUG
    )
    return _section(left_value, right_value, path_names, category)


def _mandatory_resources(item: _WorkloadArtifacts) -> dict[str, object]:
    total = _total(item)
    metadata = _metadata(item)
    resource_model = _mapping(
        metadata.get("resource_model"),
        "resource_model",
    )
    return {
        "macs": total.get("macs"),
        "macs_by_precision": total.get("macs_by_precision"),
        "resource_work": total.get("resource_work"),
        "resource_model_version": resource_model.get("version"),
    }


def _resource_section(
    left: _WorkloadArtifacts,
    right: _WorkloadArtifacts,
    path_names: tuple[str, str],
) -> ComparisonSection:
    left_value = _mandatory_resources(left)
    right_value = _mandatory_resources(right)
    category = _resource_category(left, right)
    return _section(left_value, right_value, path_names, category)


def _resource_category(
    left: _WorkloadArtifacts,
    right: _WorkloadArtifacts,
) -> DifferenceCategory:
    if _unreachable_modeled_layers(left) or _unreachable_modeled_layers(right):
        return DifferenceCategory.EXTRACTION_TOPOLOGY_LOSS
    if _dtype_profile(left) != _dtype_profile(right):
        return DifferenceCategory.NORMALIZATION_DIFFERENCE
    return DifferenceCategory.RESOURCE_MODEL_BUG


def _unreachable_modeled_layers(item: _WorkloadArtifacts) -> tuple[str, ...]:
    live = _live_layer_ids(item.graph)
    layers = _mapping(item.analysis.get("layers"), "analysis layers")
    return tuple(
        str(layer_id)
        for layer_id, raw_layer in layers.items()
        if layer_id not in live
        and bool(
            _mapping(
                _mapping(raw_layer, "analysis layer").get("resources"),
                "layer resources",
            ).get("work"),
        )
    )


def _dtype_profile(item: _WorkloadArtifacts) -> dict[str, int]:
    counts: Counter[str] = Counter()
    layers = _mapping(item.analysis.get("layers"), "analysis layers")
    for raw_layer in layers.values():
        layer = _mapping(raw_layer, "analysis layer")
        dtypes = _mapping(layer.get("tensor_dtypes"), "tensor_dtypes")
        for side in ("inputs", "outputs"):
            counts.update(
                str(value)
                for value in _list(dtypes.get(side), f"tensor_dtypes.{side}")
            )
    return dict(sorted(counts.items()))


def _fusion_intermediates(item: _WorkloadArtifacts) -> dict[str, object]:
    total = _total(item)
    return {
        key: total.get(key)
        for key in (
            "num_layers",
            "unfused_elements",
            "unfused_bytes",
            "intermediate_elements",
            "intermediate_bytes",
            "num_intermediate_tensors",
            "num_orphaned_layers",
            "fused_prefetched_elements",
            "fused_prefetched_bytes",
            "orojenesis_elements",
        )
    }


def _formal_bound(item: _WorkloadArtifacts) -> dict[str, object]:
    total = _total(item)
    metadata = _metadata(item)
    manifest_bound = _mapping(item.manifest.get("bound"), "manifest bound")
    return {
        "bound_kind": metadata.get("bound_kind"),
        "resource_seconds": total.get("resource_seconds"),
        "compute_resource": total.get("compute_resource"),
        "lower_bound_components": total.get("lower_bound_components"),
        "lower_bound_seconds": total.get("lower_bound_seconds"),
        "limiting_resource": manifest_bound.get("limiting_resource"),
        "manifest_seconds": manifest_bound.get("seconds"),
    }


def _bound_category(
    external: ComparisonSection,
    model_io: ComparisonSection,
    resources: ComparisonSection,
    fusion: ComparisonSection,
) -> DifferenceCategory:
    for section in (external, model_io, resources, fusion):
        if not section.match and section.classification is not None:
            return section.classification
    return DifferenceCategory.FORMAL_BOUND_POLICY_DIFFERENCE


__all__ = ["compare_solar_ir_paths"]

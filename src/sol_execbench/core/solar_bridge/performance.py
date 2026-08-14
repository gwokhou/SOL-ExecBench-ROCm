# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Validated SOLAR-to-performance-model boundary."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

from sol_execbench.core.bench.performance_model.models import (
    EvidenceReference,
    SemanticCharacterization,
)
from sol_execbench.core.integrity import (
    sha256_file,
    validate_relative_artifact_path,
)
from sol_execbench.core.solar_bridge.models import (
    SolarRequestManifest,
)
from sol_execbench.core.solar_bridge.semantic_descriptors import (
    fusion_regions,
    fusion_regions_cover_operations,
    semantic_descriptor,
)
from solar.artifacts import load_yaml_artifact
from solar.schema_versions import (
    SOLAR_ANALYSIS_SCHEMA_VERSION,
)


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
    descriptor, kind, reasons = semantic_descriptor(layers, metadata)
    regions = fusion_regions(metadata.get("fusion"))
    if not regions:
        reasons.append("fusion_regions_missing")
    elif not fusion_regions_cover_operations(layers, regions):
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


__all__ = [
    "load_manifest_semantic_characterization",
    "load_semantic_characterization",
]

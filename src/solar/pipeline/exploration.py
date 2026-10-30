# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Read pinned projections and adapt them to bounded exploration contracts."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from solar.analysis.demand_contracts import (
    DEMAND_ASSUMPTIONS,
    ComputationScenario,
    DemandStatus,
)
from solar.analysis.exploration_contracts import (
    MAX_EXPLORATION_DOCUMENT_BYTES,
    MAX_MODEL_MODES,
    ExplorationContract,
    FixedModel,
    ModeWork,
    ProjectionReference,
)
from solar.analysis.views import (
    AnalysisView,
    ViewAvailability,
    ViewEstimate,
    ViewReason,
    read_analysis_views,
)
from solar.artifacts.bundle import read_regular_file
from solar.artifacts.document import (
    ArtifactDocument,
    ArtifactMap,
    parse_yaml_artifact,
)
from solar.ir.access import AccessQuality
from solar.pipeline.reuse_artifacts import load_projection
from solar.reuse_contracts import ReuseResult
from solar.rocm.architecture import ArchitectureProfile
from solar.schema_versions import SOLAR_EXPLORATION_SCHEMA_VERSION


def load_exploration_request[T: ExplorationContract](
    path: Path, contract: type[T]
) -> T:
    """Reject incompatible versions and oversized documents before field parsing."""
    raw = parse_yaml_artifact(
        path, read_regular_file(path, max_bytes=MAX_EXPLORATION_DOCUMENT_BYTES)
    )
    if (
        type(raw.get("schema_version")) is not int
        or raw.get("schema_version") != SOLAR_EXPLORATION_SCHEMA_VERSION
    ):
        raise ValueError("exploration schema is not current")
    return contract.model_validate(raw.data)


def read_projection(source: ProjectionReference) -> ReuseResult:
    """Reuse the exact pinned artifact reader; no reference or graph execution."""
    return load_projection(
        source.directory,
        record_sha256=source.record_sha256,
        projection_sha256=source.projection_sha256,
        budget=source.budget,
    )


def model_from_projection(
    result: ReuseResult,
    view: AnalysisView,
    computation: ComputationScenario = ComputationScenario.DENSE,
) -> FixedModel:
    """Adapt an owned, accepted result; bound the modes before building new objects."""
    document = ArtifactDocument(result.output_dir, result.analysis)
    metadata = ArtifactDocument(
        result.output_dir, document.require_mapping("metadata")
    )
    profile = ArchitectureProfile.load(metadata.require_mapping("architecture"))
    views = projection_views(result, computation)
    total = ArtifactDocument(
        result.output_dir, document.require_mapping("total")
    )
    raw = ArtifactDocument(
        result.output_dir,
        cast(ArtifactMap, result.restricted.resource_work)
        if computation is ComputationScenario.RESTRICTED
        and result.restricted is not None
        else total.require_mapping("resource_work"),
    )
    if len(raw) > MAX_MODEL_MODES:
        raise ValueError("resource model exceeds mode budget")
    resources: dict[str, dict[str, ModeWork]] = {}
    count = 0
    for name in raw:
        modes = raw.require_mapping(name)
        count += len(modes)
        if count > MAX_MODEL_MODES:
            raise ValueError("resource model exceeds mode budget")
        resources[name] = {
            mode: ModeWork.model_validate(
                {"work": value, "rate": profile.resource_rate_for(name, mode)}
            )
            for mode, value in modes.items()
        }
    return FixedModel(
        projection_sha256=result.record.projection_sha256,
        view=view,
        computation=computation,
        estimate=views[view],
        resources=resources,
        bandwidth=profile.memory_bandwidth_bytes_per_second,
    )


def projection_views(
    result: ReuseResult, computation: ComputationScenario
) -> dict[AnalysisView, ViewEstimate]:
    """Select explicitly without replacing an unavailable restricted view with dense."""
    if computation is ComputationScenario.DENSE:
        metadata = ArtifactDocument(
            result.output_dir, result.analysis
        ).require_mapping("metadata")
        return read_analysis_views(metadata.get("views"))
    restricted = result.restricted
    if restricted is None:
        raise ValueError("projection has no restricted computation scenario")
    if restricted.status is DemandStatus.AVAILABLE:
        return restricted.views
    return {
        kind: ViewEstimate(
            status=ViewAvailability.UNAVAILABLE,
            reason=ViewReason.RESTRICTED_UNAVAILABLE,
            seconds=None,
            io_bytes=None,
            resource_seconds={},
            operator_seconds={},
            read_quality=AccessQuality.UNAVAILABLE,
            assumptions=(*DEMAND_ASSUMPTIONS, restricted.reason.value),
        )
        for kind in AnalysisView
    }

# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Scan identity and resume acceptance over recursively pinned projections."""

from __future__ import annotations

from solar.analysis.identity import analysis_rules_identity
from solar.analysis.orojenesis.configuration import (
    orojenesis_mapper_thread_count,
)
from solar.analysis.scan_contracts import (
    MAX_SCAN_POINTS,
    MAX_SCAN_REPORT_BYTES,
    SCAN_FILENAME,
    SCAN_WORKER_GRACE_SECONDS,
    PointStatus,
    ScanPoint,
    ScanReference,
    ScanResult,
)
from solar.analysis.scan_grid import profile_at_point
from solar.analysis.views import ViewAvailability
from solar.artifacts import sha256_bytes, stable_json_checksum
from solar.artifacts.bundle import read_regular_file
from solar.artifacts.document import parse_yaml_artifact
from solar.errors import ToolchainUnavailableError
from solar.pipeline.exploration import projection_views, read_projection
from solar.pipeline.reuse_projection import configure_projection
from solar.reuse_contracts import ProfileScenario, ReuseResult
from solar.rocm.architecture import ArchitectureProfile
from solar.scan_contracts import ScanRequest
from solar.schema_versions import SOLAR_EXPLORATION_SCHEMA_VERSION


def scan_identity(
    request: ScanRequest,
    baseline: ArchitectureProfile,
    execution_identity: dict[str, str],
) -> dict[str, object]:
    """Bind every parameter, scheduler budget and currently accepted mapper identity."""
    job = request.job.model_dump(mode="json")
    job["projection"].pop("output_dir")
    job["projection"]["architecture"] = baseline.to_dict()
    job["projection"]["hypothetical"] = True
    template = request.job.projection.model_copy(
        update={"architecture": baseline.to_dict(), "hypothetical": True}
    )
    try:
        mapper = configure_projection(template, baseline.to_dict()).identity[
            "mapper"
        ]
    except ToolchainUnavailableError as exc:
        mapper = {"unavailable": exc.reason_code, "message": str(exc)[:4096]}
    return {
        "job": job,
        "axes": [axis.model_dump(mode="json") for axis in request.axes],
        "view": request.view.value,
        "computation": request.computation.value,
        "budget": request.budget.model_dump(mode="json"),
        "objectives": [
            item.model_dump(mode="json") for item in request.objectives
        ],
        "target_seconds": request.target_seconds,
        "rules": analysis_rules_identity(),
        "mapper": mapper,
        "mapper_threads": orojenesis_mapper_thread_count(),
        "concurrency": 1,
        "outer_grace_seconds": SCAN_WORKER_GRACE_SECONDS,
        "execution": execution_identity,
    }


def load_scan(reference: ScanReference) -> ScanResult:
    """Accept a pinned terminal report; incomplete progress is not a cache hit."""
    path = reference.directory / SCAN_FILENAME
    payload = read_regular_file(path, max_bytes=MAX_SCAN_REPORT_BYTES)
    if sha256_bytes(payload) != reference.record_sha256:
        raise ValueError("scan record SHA-256 differs from pin")
    document = parse_yaml_artifact(path, payload)
    if (
        type(document.get("schema_version")) is not int
        or document.get("schema_version") != SOLAR_EXPLORATION_SCHEMA_VERSION
    ):
        raise ValueError("scan schema is not current")
    raw_points = document.get("points")
    if (
        not isinstance(raw_points, list)
        or not 1 <= len(raw_points) <= MAX_SCAN_POINTS
    ):
        raise ValueError("scan record exceeds point budget")
    result = ScanResult.model_validate(document.data)
    if stable_json_checksum(result.identity_inputs) != result.scan_sha256:
        raise ValueError("scan identity checksum differs")
    if result.baseline_point_id != result.points[0].point_id or any(
        point.index != index for index, point in enumerate(result.points)
    ):
        raise ValueError("scan point order or baseline differs")
    if len({point.point_id for point in result.points}) != len(result.points):
        raise ValueError("scan contains duplicate points")
    return result


def check_projection(
    point: ScanPoint,
    result: ReuseResult,
    request: ScanRequest,
    baseline: ArchitectureProfile,
    identity: dict[str, object],
) -> ScanPoint:
    """Verify that a completed result belongs to this exact planned point."""
    expected_profile = profile_at_point(baseline, request, point.coordinates)
    record = result.record
    projection_inputs = record.identity_inputs["projection"]
    if not isinstance(projection_inputs, dict):
        raise ValueError("projection identity is malformed")
    if (
        record.source_manifest_sha256 != request.job.source.manifest_sha256
        or record.requested_architecture_sha256
        != stable_json_checksum(expected_profile.to_dict())
        or record.rules != identity["rules"]
        or projection_inputs.get("mapper") != identity["mapper"]
        or projection_inputs.get("restricted")
        != (
            request.job.projection.restricted.model_dump(mode="json")
            if request.job.projection.restricted is not None
            else None
        )
        or record.scenario is not ProfileScenario.HYPOTHETICAL
        or record.sol_score_eligible
        or record.publication_eligible
        or record.profile_audit_verified
    ):
        raise ValueError(
            "scan projection source/profile/rules/mapper/qualification differs"
        )
    views = projection_views(result, request.computation)
    estimate = views[request.view]
    memory_seconds = (
        estimate.io_bytes / expected_profile.memory_bandwidth_bytes_per_second
        if estimate.io_bytes is not None
        else None
    )
    status = (
        PointStatus.ANALYZED
        if estimate.status is ViewAvailability.AVAILABLE
        else PointStatus.UNAVAILABLE
    )
    if point.result is not None and (
        point.estimate != estimate
        or point.status is not status
        or point.views != views
        or point.memory_seconds != memory_seconds
    ):
        raise ValueError("cached scan estimate disagrees with bound analysis")
    return point.model_copy(
        update={
            "estimate": estimate,
            "views": views,
            "memory_seconds": memory_seconds,
            "status": status,
            "reason": estimate.reason.value,
        }
    )


def resume_points(
    request: ScanRequest,
    planned: tuple[ScanPoint, ...],
    baseline: ArchitectureProfile,
    identity: dict[str, object],
) -> dict[str, ScanPoint]:
    """Revalidate complete prior points without overwriting or retrying old directories."""
    if request.resume is None:
        return {}
    previous = load_scan(request.resume)
    if previous.scan_sha256 != stable_json_checksum(identity) or len(
        previous.points
    ) != len(planned):
        raise ValueError("resume scan identity differs")
    cached = {}
    for old, point in zip(previous.points, planned, strict=True):
        if (
            old.point_id != point.point_id
            or old.coordinates != point.coordinates
        ):
            raise ValueError("resume grid point differs")
        if old.result is None:
            continue
        accepted = check_projection(
            old, read_projection(old.result), request, baseline, identity
        )
        cached[point.point_id] = accepted.model_copy(
            update={"resumed": True, "elapsed_seconds": 0.0}
        )
    return cached

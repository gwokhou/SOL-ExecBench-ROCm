# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Serial, bounded fixed-instance exploration with immutable attempt outputs."""

from __future__ import annotations

from time import monotonic

from solar.analysis.exploration_contracts import ProjectionReference
from solar.analysis.interrupts import (
    AnalysisDeadline,
    analysis_deadline,
    cleanup_on_termination,
)
from solar.analysis.scan_contracts import (
    MAX_SCAN_MESSAGE_CHARS,
    MAX_SCAN_REPORT_BYTES,
    SCAN_FILENAME,
    SCAN_PLAN_FILENAME,
    PointStatus,
    ScanPoint,
    ScanResult,
    ScanStopReason,
)
from solar.analysis.scan_grid import plan_grid, profile_at_point
from solar.artifacts import sha256_file, stable_json_checksum
from solar.artifacts.json import write_json_artifact
from solar.artifacts.names import REUSE_FILENAME
from solar.errors import SolarError, ToolchainUnavailableError
from solar.pipeline.reuse_projection import load_projection_profile
from solar.pipeline.scan_artifacts import (
    check_projection,
    resume_points,
    scan_identity,
)
from solar.reuse import ReuseSession
from solar.scan_contracts import ScanRequest
from solar.schema_versions import SOLAR_EXPLORATION_SCHEMA_VERSION

MAX_EXECUTION_ID_FIELDS = 16
MAX_EXECUTION_ID_NAME_CHARS = 128
MAX_EXECUTION_ID_VALUE_CHARS = 256


class ScanExecutor:
    """Own one fixed plan and one count snapshot; points never run concurrently."""

    def __init__(
        self, request: ScanRequest, execution_identity: dict[str, str]
    ) -> None:
        """Validate the finite plan and reserve a new output directory."""
        self.started = monotonic()
        if len(execution_identity) > MAX_EXECUTION_ID_FIELDS or any(
            not isinstance(key, str)
            or not isinstance(value, str)
            or len(key) > MAX_EXECUTION_ID_NAME_CHARS
            or len(value) > MAX_EXECUTION_ID_VALUE_CHARS
            for key, value in execution_identity.items()
        ):
            raise ValueError("scan execution identity exceeds its field budget")
        self.request = request
        self.baseline = load_projection_profile(
            request.job.projection.architecture
        )
        self.planned = plan_grid(request, self.baseline)
        self.identity = scan_identity(
            request, self.baseline, execution_identity
        )
        self.output = request.job.projection.output_dir.resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.points = list(self.planned)
        self.stop_reason = ScanStopReason.INTERRUPTED
        write_json_artifact(
            self.output / SCAN_PLAN_FILENAME,
            {
                "schema_version": SOLAR_EXPLORATION_SCHEMA_VERSION,
                "identity_inputs": self.identity,
                "points": [
                    point.model_dump(mode="json") for point in self.points
                ],
            },
            max_bytes=MAX_SCAN_REPORT_BYTES,
        )

    def run(self) -> ScanResult:
        """Record all planned terminal states even when preparation is interrupted."""
        try:
            with cleanup_on_termination():
                remaining = self._remaining()
                if remaining <= 0:
                    self.stop_reason = ScanStopReason.TOTAL_BUDGET
                else:
                    self._run_prepared(remaining)
        finally:
            self.points = [
                point.model_copy(update={"reason": self.stop_reason})
                if point.reason == "not_started"
                else point
                for point in self.points
            ]
            result = ScanResult(
                scan_sha256=stable_json_checksum(self.identity),
                identity_inputs=self.identity,
                baseline_point_id=self.planned[0].point_id,
                points=tuple(self.points),
                elapsed_seconds=monotonic() - self.started,
                stop_reason=self.stop_reason,
                previous=self.request.resume,
            )
            write_json_artifact(
                self.output / SCAN_FILENAME,
                result.model_dump(mode="json"),
                max_bytes=MAX_SCAN_REPORT_BYTES,
            )
        return result

    def _remaining(self) -> float:
        return self.request.budget.total_seconds - (monotonic() - self.started)

    def _run_prepared(self, remaining: float) -> None:
        try:
            with analysis_deadline(remaining):
                cached = resume_points(
                    self.request, self.planned, self.baseline, self.identity
                )
                session = ReuseSession(
                    self.request.job.source,
                    access_budget=self.request.job.access_budget,
                )
        except AnalysisDeadline:
            self.stop_reason = ScanStopReason.PREPARATION_BUDGET
            return
        for point in self.planned:
            if point.point_id in cached:
                self.points[point.index] = cached[point.point_id]
                continue
            if self._remaining() <= 0:
                self.stop_reason = ScanStopReason.TOTAL_BUDGET
                break
            self._attempt(session, point)
            if self.stop_reason is ScanStopReason.TOTAL_BUDGET:
                break
        else:
            self.stop_reason = ScanStopReason.COMPLETED

    def _attempt(self, session: ReuseSession, point: ScanPoint) -> None:
        started = monotonic()
        remaining = self._remaining()
        if remaining <= 0:
            self.stop_reason = ScanStopReason.TOTAL_BUDGET
            return
        current = point.model_copy(
            update={
                "attempted": True,
                "status": PointStatus.FAILED,
                "reason": "invariant_failure",
            }
        )
        self.points[point.index] = current
        try:
            with analysis_deadline(
                min(self.request.budget.point_seconds, remaining)
            ):
                current = self._project(session, point)
        except AnalysisDeadline:
            current = current.model_copy(
                update={
                    "status": PointStatus.TIMEOUT,
                    "reason": "point_deadline",
                }
            )
        except ToolchainUnavailableError as exc:
            current = current.model_copy(
                update={
                    "status": PointStatus.FAILED,
                    "reason": exc.reason_code,
                    "message": str(exc)[:MAX_SCAN_MESSAGE_CHARS],
                }
            )
        except ValueError as exc:
            current = current.model_copy(
                update={
                    "status": PointStatus.INVALID,
                    "reason": exc.reason_code
                    if isinstance(exc, SolarError)
                    else "invalid_point",
                    "message": str(exc)[:MAX_SCAN_MESSAGE_CHARS],
                }
            )
        except OSError as exc:
            current = current.model_copy(
                update={
                    "status": PointStatus.FAILED,
                    "reason": "point_io_failure",
                    "message": str(exc)[:MAX_SCAN_MESSAGE_CHARS],
                }
            )
        except SystemExit:
            current = current.model_copy(
                update={
                    "status": PointStatus.TIMEOUT,
                    "reason": "worker_terminated",
                }
            )
            raise
        finally:
            current = current.model_copy(
                update={"elapsed_seconds": monotonic() - started}
            )
            self.points[point.index] = current
            write_json_artifact(
                self.output / f"{point.point_id}.json",
                current.model_dump(mode="json"),
                max_bytes=MAX_SCAN_REPORT_BYTES,
            )

    def _project(self, session: ReuseSession, point: ScanPoint) -> ScanPoint:
        profile = profile_at_point(
            self.baseline, self.request, point.coordinates
        )
        projection = self.request.job.projection.model_copy(
            update={
                "architecture": profile.to_dict(),
                "hypothetical": True,
                "output_dir": self.output / "points" / point.point_id,
            }
        )
        result = session.project(projection)
        accepted = check_projection(
            point, result, self.request, self.baseline, self.identity
        )
        return accepted.model_copy(
            update={
                "attempted": True,
                "result": ProjectionReference(
                    directory=result.output_dir,
                    record_sha256=sha256_file(
                        result.output_dir / REUSE_FILENAME
                    ),
                    projection_sha256=result.record.projection_sha256,
                    budget=self.request.job.source.budget,
                ),
            }
        )

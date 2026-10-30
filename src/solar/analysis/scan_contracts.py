# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Finite architecture grids and immutable attempt records."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from solar.analysis.demand_contracts import ComputationScenario
from solar.analysis.exploration_contracts import (
    ExplorationContract,
    FinitePositive,
    ProjectionReference,
)
from solar.analysis.views import (
    AnalysisView,
    FiniteNonnegative,
    ViewAvailability,
    ViewEstimate,
)
from solar.artifacts.checksums import SHA256

MAX_SCAN_POINTS = 64
MAX_SCAN_AXES = 8
MAX_SCAN_REPORT_BYTES = 8 * 1024 * 1024
MAX_SCAN_MESSAGE_CHARS = 4096
SCAN_FILENAME = "scan.json"
SCAN_PLAN_FILENAME = "plan.json"
DEFAULT_SCAN_SECONDS = 300.0
DEFAULT_POINT_SECONDS = 30.0
SCAN_WORKER_GRACE_SECONDS = 10.0
type AxisValue = Annotated[int, Field(strict=True, gt=0)] | FinitePositive


class AxisKind(StrEnum):
    """Supported fixed-instance architecture changes."""

    RESOURCE_SCALE = "resource_scale"
    BANDWIDTH_SCALE = "bandwidth_scale"
    CAPACITY_BYTES = "capacity_bytes"


class ScanAxis(BaseModel):
    """One axis, starting with the unmodified baseline coordinate."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: AxisKind
    values: Annotated[
        tuple[AxisValue, ...], Field(min_length=1, max_length=MAX_SCAN_POINTS)
    ]
    resources: tuple[str, ...] = ()
    memory_level: str | None = None

    @model_validator(mode="after")
    def _consistent_axis(self) -> ScanAxis:
        if len(set(self.values)) != len(self.values):
            raise ValueError("scan axis values must be unique")
        if self.kind is AxisKind.RESOURCE_SCALE:
            if (
                not self.resources
                or len(set(self.resources)) != len(self.resources)
                or self.memory_level is not None
            ):
                raise ValueError(
                    "resource axis requires a distinct nonempty resource group"
                )
        elif self.resources:
            raise ValueError("only resource axes select resource groups")
        if self.kind is AxisKind.CAPACITY_BYTES:
            if (
                not self.memory_level
                or self.memory_level == "vram"
                or any(type(value) is not int for value in self.values)
            ):
                raise ValueError(
                    "capacity axis requires a cache level and positive integer bytes"
                )
        elif self.memory_level is not None or self.values[0] != 1:
            raise ValueError(
                "scale axes must begin at one and cannot select a cache"
            )
        return self

    @property
    def key(self) -> str:
        """Return a deterministic label, never a filesystem path."""
        suffix = (
            ",".join(sorted(self.resources))
            if self.resources
            else self.memory_level or ""
        )
        return f"{self.kind}:{suffix}"


class ScanBudget(BaseModel):
    """Per-attempt wall-time and planned-work limits; no concurrency fanout."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    max_points: Annotated[int, Field(strict=True, gt=0, le=MAX_SCAN_POINTS)] = (
        MAX_SCAN_POINTS
    )
    total_seconds: FinitePositive = DEFAULT_SCAN_SECONDS
    point_seconds: FinitePositive = DEFAULT_POINT_SECONDS


class ScanReference(BaseModel):
    """The caller's pin for a previous attempt; old directories remain intact."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    directory: Path
    record_sha256: SHA256


class ObjectiveKind(StrEnum):
    """Explicit comparable quantities; none is a price, area or power model."""

    SECONDS = "model_seconds"
    AXIS = "axis_value"


class ScanObjective(BaseModel):
    """A quantity to minimize in an optional Pareto comparison."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: ObjectiveKind
    axis: str | None = None

    @model_validator(mode="after")
    def _axis_required(self) -> ScanObjective:
        if (self.kind is ObjectiveKind.AXIS) != (self.axis is not None):
            raise ValueError("only axis objectives require an axis key")
        return self


class ScanDefinition(ExplorationContract):
    """Fixed source, ordered grid, budgets and an optional pinned prior attempt."""

    axes: Annotated[tuple[ScanAxis, ...], Field(max_length=MAX_SCAN_AXES)] = ()
    view: AnalysisView
    computation: ComputationScenario = ComputationScenario.DENSE
    budget: ScanBudget = ScanBudget()
    objectives: Annotated[
        tuple[ScanObjective, ...], Field(max_length=MAX_SCAN_AXES + 1)
    ] = ()
    target_seconds: FinitePositive | None = None

    @model_validator(mode="before")
    @classmethod
    def _bound_grid(cls, raw: object) -> object:
        if not isinstance(raw, Mapping):
            return raw
        axes = raw.get("axes", ())
        if not isinstance(axes, (tuple, list)):
            return raw
        if len(axes) > MAX_SCAN_AXES:
            raise ValueError("scan exceeds axis budget")
        budget = raw.get("budget", {})
        maximum = (
            budget.get("max_points", MAX_SCAN_POINTS)
            if isinstance(budget, Mapping)
            else budget.max_points
            if isinstance(budget, ScanBudget)
            else MAX_SCAN_POINTS
        )
        if type(maximum) is not int or not 0 < maximum <= MAX_SCAN_POINTS:
            raise ValueError("invalid scan point budget")
        cardinality = 1
        for axis in axes:
            values = (
                axis.get("values", ())
                if isinstance(axis, Mapping)
                else axis.values
                if isinstance(axis, ScanAxis)
                else ()
            )
            if isinstance(values, (tuple, list)):
                cardinality *= len(values)
                if cardinality > maximum:
                    raise ValueError("scan grid exceeds point budget")
        return raw

    @model_validator(mode="after")
    def _distinct_axes(self) -> ScanDefinition:
        keys = [axis.key for axis in self.axes]
        resources = [name for axis in self.axes for name in axis.resources]
        if len(keys) != len(set(keys)) or len(resources) != len(set(resources)):
            raise ValueError("scan axes must not overlap")
        objectives = [(item.kind, item.axis) for item in self.objectives]
        if len(objectives) != len(set(objectives)) or len(objectives) == 1:
            raise ValueError(
                "Pareto comparison requires distinct multiple objectives"
            )
        if any(
            item.axis not in keys
            for item in self.objectives
            if item.kind is ObjectiveKind.AXIS
        ):
            raise ValueError("Pareto objective names an unknown axis")
        return self


class PointStatus(StrEnum):
    """Terminal states include missing views, invalid inputs and unfinished work."""

    ANALYZED = "analyzed"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"
    INVALID = "invalid"
    TIMEOUT = "timeout"
    NOT_RUN = "not_run"


class ScanStopReason(StrEnum):
    """Why an attempt finished scheduling new points."""

    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    TOTAL_BUDGET = "total_budget_exhausted"
    PREPARATION_BUDGET = "total_budget_exhausted_during_preparation"


class ScanPoint(BaseModel):
    """All planned points remain visible, including failures and budget stops."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    index: Annotated[int, Field(strict=True, ge=0)]
    point_id: Annotated[str, Field(pattern=r"^p[0-9]{4}-[0-9a-f]{12}$")]
    coordinates: dict[str, AxisValue]
    status: PointStatus
    reason: str
    message: Annotated[str, Field(max_length=MAX_SCAN_MESSAGE_CHARS)] | None = (
        None
    )
    attempted: bool = False
    elapsed_seconds: FiniteNonnegative = 0.0
    result: ProjectionReference | None = None
    estimate: ViewEstimate | None = None
    views: dict[AnalysisView, ViewEstimate] | None = None
    memory_seconds: FiniteNonnegative | None = None
    resumed: bool = False

    @model_validator(mode="after")
    def _complete_result(self) -> ScanPoint:
        completed = self.status in {
            PointStatus.ANALYZED,
            PointStatus.UNAVAILABLE,
        }
        if completed != (self.result is not None and self.estimate is not None):
            raise ValueError(
                "complete scan point requires a bound projection and view"
            )
        if not completed and (
            self.result is not None or self.estimate is not None
        ):
            raise ValueError("failed point cannot carry an accepted result")
        if completed and (
            self.views is None or set(self.views) != set(AnalysisView)
        ):
            raise ValueError("complete point requires all three views")
        if not completed and (
            self.views is not None or self.memory_seconds is not None
        ):
            raise ValueError("failed point cannot carry numeric estimates")
        if self.status is PointStatus.ANALYZED and self.memory_seconds is None:
            raise ValueError("analyzed point requires memory time")
        if completed and self.estimate is not None:
            if (self.status is PointStatus.ANALYZED) != (
                self.estimate.status is ViewAvailability.AVAILABLE
            ):
                raise ValueError("point status disagrees with selected view")
            if (
                self.status is PointStatus.UNAVAILABLE
                and self.memory_seconds is not None
            ):
                raise ValueError("unavailable point cannot carry memory time")
        return self


class ScanResult(ExplorationContract):
    """One immutable scan attempt; resume references accepted prior artifacts."""

    scan_sha256: SHA256
    identity_inputs: dict[str, object]
    baseline_point_id: str
    points: Annotated[
        tuple[ScanPoint, ...], Field(min_length=1, max_length=MAX_SCAN_POINTS)
    ]
    elapsed_seconds: FiniteNonnegative
    stop_reason: ScanStopReason
    previous: ScanReference | None = None
    performance_authority: Literal[False] = False
    publication_eligible: Literal[False] = False

# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Versioned, bounded contraction-search plans and complete candidate states."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from solar.analysis.exploration_contracts import (
    FinitePositive,
    ProjectionReference,
)
from solar.analysis.order_domain import MatrixChain
from solar.analysis.order_trees import (
    MAX_ORDER_CANDIDATES,
    MAX_ORDER_LEAVES,
    OrderCost,
)
from solar.analysis.views import AnalysisView, FiniteNonnegative, ViewEstimate
from solar.artifacts.checksums import SHA256, stable_json_checksum
from solar.schema_versions import SOLAR_ORDER_SCHEMA_VERSION

ORDER_FILENAME = "order-search.json"
ORDER_PLAN_FILENAME = "order-plan.json"
ORDER_CANDIDATE_FILENAME = "candidate.json"
MAX_ORDER_REPORT_BYTES = 8 * 1024 * 1024
MAX_ORDER_MESSAGE_CHARS = 4096
MAX_ORDER_ELEMENTS = 1_000_000
MAX_ORDER_MACS = 10_000_000
DEFAULT_ORDER_SECONDS = 300.0
DEFAULT_CANDIDATE_SECONDS = 30.0
ORDER_WORKER_GRACE_SECONDS = 10.0
MAX_ORDER_EXECUTION_FIELDS = 16
MAX_ORDER_EXECUTION_KEY_CHARS = 128
MAX_ORDER_EXECUTION_VALUE_CHARS = 256
ORDER_IDENTITY_FIELDS = frozenset(
    {
        "job",
        "chain",
        "reference_name",
        "trace_seed",
        "precision",
        "view",
        "computation",
        "budget",
        "rules",
        "execution",
        "profile",
        "mapper",
        "algebra",
        "tie_rule",
        "concurrency",
    }
)
type TreeTokens = Annotated[
    tuple[int, ...], Field(min_length=3, max_length=2 * MAX_ORDER_LEAVES - 1)
]


class OrderContract(BaseModel):
    """The exact current search schema precedes every external business field."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Annotated[int, Field(strict=True)] = (
        SOLAR_ORDER_SCHEMA_VERSION
    )

    @model_validator(mode="before")
    @classmethod
    def current_version(cls, raw: object) -> object:
        """Reject superseded versions before parsing their contents."""
        if isinstance(raw, Mapping):
            version = raw.get("schema_version", SOLAR_ORDER_SCHEMA_VERSION)
            if (
                type(version) is not int
                or version != SOLAR_ORDER_SCHEMA_VERSION
            ):
                raise ValueError("contraction search schema is not current")
        return raw


class OrderBudget(BaseModel):
    """Actual materialization/work limits plus serial candidate and wall budgets."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    max_candidates: Annotated[
        int, Field(strict=True, gt=0, le=MAX_ORDER_CANDIDATES)
    ] = MAX_ORDER_CANDIDATES
    total_seconds: FinitePositive = DEFAULT_ORDER_SECONDS
    candidate_seconds: FinitePositive = DEFAULT_CANDIDATE_SECONDS
    max_elements: Annotated[
        int, Field(strict=True, gt=0, le=MAX_ORDER_ELEMENTS)
    ] = MAX_ORDER_ELEMENTS
    max_macs: Annotated[int, Field(strict=True, gt=0, le=MAX_ORDER_MACS)] = (
        MAX_ORDER_MACS
    )


class CandidateStatus(StrEnum):
    """A failed or unevaluated candidate never receives a numerical score."""

    ANALYZED = "analyzed"
    UNAVAILABLE = "view_unavailable"
    VERIFICATION_FAILED = "verification_failed"
    FAILED = "failed"
    WORK_BUDGET = "work_budget_exhausted"
    TIMEOUT = "timeout"
    NOT_RUN = "not_run"


class OrderStopReason(StrEnum):
    """Scheduling completion is distinct from verification and global coverage."""

    COMPLETED = "completed"
    CANDIDATE_BUDGET = "candidate_budget_exhausted"
    TOTAL_BUDGET = "total_budget_exhausted"
    INTERRUPTED = "interrupted"
    IDENTITY_CHANGED = "rules_changed_during_search"


class OrderSelectionReason(StrEnum):
    """The selected result never implies optimality outside the verified domain."""

    NONE = "no_verified_available_candidate"
    MINIMUM = "minimum_over_exhaustive_verified_domain"
    BEST_AVAILABLE = "best_available_with_incomplete_coverage"


class OrderReference(BaseModel):
    """A caller-pinned immutable search attempt or candidate bundle."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    directory: Path
    record_sha256: SHA256


class OrderCandidate(BaseModel):
    """One association including failures, its original-case flag and all views."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    candidate_id: Annotated[str, Field(pattern=r"^c-[0-9a-f]{16}$")]
    tree: TreeTokens
    cost: OrderCost
    original: bool
    status: CandidateStatus = CandidateStatus.NOT_RUN
    reason: str = "not_started"
    message: (
        Annotated[str, Field(max_length=MAX_ORDER_MESSAGE_CHARS)] | None
    ) = None
    elapsed_seconds: FiniteNonnegative = 0.0
    attempted: bool = False
    resumed: bool = False
    bundle: OrderReference | None = None
    projection: ProjectionReference | None = None
    views: dict[AnalysisView, ViewEstimate] | None = None

    @model_validator(mode="after")
    def _complete_views(self) -> OrderCandidate:
        complete = self.status in {
            CandidateStatus.ANALYZED,
            CandidateStatus.UNAVAILABLE,
        }
        if complete != (self.projection is not None and self.views is not None):
            raise ValueError(
                "only completed candidates carry a projection and views"
            )
        if self.views is not None and set(self.views) != set(AnalysisView):
            raise ValueError("candidate must retain all three views")
        if not complete and (
            self.projection is not None or self.views is not None
        ):
            raise ValueError("failed candidate cannot carry a numeric estimate")
        return self


class CandidateRecord(OrderContract):
    """Complete inventory binds both rejected artifacts and accepted projections."""

    search_sha256: SHA256
    candidate: OrderCandidate
    artifacts: dict[str, SHA256]


class OrderResult(OrderContract):
    """The complete denominator; optimality is limited to this declared domain."""

    search_sha256: SHA256
    identity_inputs: dict[str, object]
    chain: MatrixChain
    baseline: ProjectionReference | None
    baseline_views: dict[AnalysisView, ViewEstimate] | None
    total_candidates: Annotated[int, Field(strict=True, gt=0)]
    candidates: Annotated[
        tuple[OrderCandidate, ...],
        Field(min_length=1, max_length=MAX_ORDER_CANDIDATES),
    ]
    unexplored_candidates: Annotated[int, Field(strict=True, ge=0)]
    selected_candidate_id: str | None
    selection_reason: OrderSelectionReason
    exhaustive_verified_domain: bool
    elapsed_seconds: FiniteNonnegative
    stop_reason: OrderStopReason
    previous: OrderReference | None = None
    performance_authority: Literal[False] = False
    publication_eligible: Literal[False] = False
    sol_score_eligible: Literal[False] = False

    @model_validator(mode="after")
    def _complete_identity(self) -> OrderResult:
        identity = self.identity_inputs
        if set(identity) != ORDER_IDENTITY_FIELDS:
            raise ValueError("search identity fields are incomplete or unknown")
        if stable_json_checksum(identity["chain"]) != stable_json_checksum(
            asdict(self.chain)
        ):
            raise ValueError("search report chain differs from its identity")
        if (
            type(identity["trace_seed"]) is not int
            or not isinstance(identity["reference_name"], str)
            or not isinstance(identity["precision"], str)
            or identity["tie_rule"] != "original_then_enumeration_order"
            or type(identity["concurrency"]) is not int
            or identity["concurrency"] != 1
        ):
            raise ValueError(
                "search identity has an invalid execution contract"
            )
        return self

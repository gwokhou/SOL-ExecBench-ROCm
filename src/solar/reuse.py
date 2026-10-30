# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Public verified-program reuse without reference execution or retracing.

Construct a ReuseSession from a pinned VerifiedProgramRequest. Project any
number of explicitly selected profiles into distinct output directories. The
session owns validated IR and intrinsic counts; it never restores Python code.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict
from functools import cache, cached_property, partial
from pathlib import Path
from tempfile import TemporaryDirectory

from solar.analysis.access_regions import DEFAULT_ACCESS_BUDGET, AccessBudget
from solar.analysis.demand import DemandGraph, restrict_computation
from solar.analysis.demand_contracts import (
    DEMAND_FILENAME,
    DemandReason,
    DemandRequest,
    DemandResult,
    DemandStatus,
)
from solar.analysis.demand_plan import DemandDeclinedError
from solar.analysis.demand_reporting import demand_report
from solar.analysis.graph_analyzer import IRGraphAnalyzer
from solar.analysis.graph_context import IntrinsicGraph
from solar.analysis.graph_loading import prepare_validated_analysis
from solar.analysis.identity import RuleGroup, analysis_rules_identity
from solar.artifacts import stable_json_checksum
from solar.artifacts.json import write_json_artifact
from solar.pipeline.reuse_artifacts import seal_projection
from solar.pipeline.reuse_projection import configure_projection
from solar.pipeline.verified_program import (
    VerifiedProgram,
    load_verified_program,
)
from solar.reuse_contracts import (
    ProjectionRequest,
    ReuseArtifactError,
    ReuseOutputExistsError,
    ReuseResult,
    ReuseRulesChangedError,
    VerifiedProgramRequest,
)
from solar.rocm.architecture import ArchitectureProfile
from solar.types import NodeDict


def _count_restricted(
    intrinsic: IntrinsicGraph, request: DemandRequest
) -> DemandGraph | DemandReason | None:
    try:
        return restrict_computation(intrinsic, request)
    except DemandDeclinedError as exc:
        return exc.reason


class ReuseSession:
    """Validated program and once-computed intrinsic work for profile exploration."""

    def __init__(
        self,
        source: VerifiedProgramRequest,
        *,
        access_budget: AccessBudget = DEFAULT_ACCESS_BUDGET,
    ) -> None:
        """Verify a complete bundle and count it without executing a reference."""
        try:
            program = load_verified_program(source)
        except (OSError, ValueError) as exc:
            raise ReuseArtifactError(
                f"cannot reuse pinned bundle: {exc}"
            ) from exc
        self._initialize(program, access_budget)

    @classmethod
    def from_verified_program(
        cls,
        program: VerifiedProgram,
        *,
        access_budget: AccessBudget = DEFAULT_ACCESS_BUDGET,
    ) -> ReuseSession:
        """Consume an owned loader result inside a workflow without rereading it."""
        session = cls.__new__(cls)
        session._initialize(program, access_budget)
        return session

    def _initialize(
        self, program: VerifiedProgram, access_budget: AccessBudget
    ) -> None:
        source = program.identity
        self._program = program
        self._rules = analysis_rules_identity()
        self._analyzer = IRGraphAnalyzer()
        graph_sha256 = next(
            item.sha256
            for item in program.manifest.artifacts
            if item.path == source.ir_path.graph_filename
        )
        prepared = prepare_validated_analysis(
            source=program.graph.path,
            source_sha256=graph_sha256,
            graph=program.graph.data,
            access=program.access,
            output_dir=Path(),
            precision=program.manifest.analysis_contract.precision,
            strict=True,
            semantic_complete=True,
        )
        self._intrinsic = self._analyzer.count_prepared(
            prepared, access_budget=access_budget
        )
        self._intrinsic_identity = {
            "source_manifest_sha256": source.manifest_sha256,
            "graph_sha256": graph_sha256,
            "verification_execution_sha256": source.execution_sha256,
            "precision": prepared.requested_precision,
            "rules_sha256": self._rules[RuleGroup.COUNTS],
            "access_budget": asdict(access_budget),
            "bundle_budget": source.budget.model_dump(mode="json"),
        }

    @property
    def intrinsic_sha256(self) -> str:
        """Return the identity of the owned fixed-instance count snapshot."""
        return stable_json_checksum(self._intrinsic_identity)

    @cached_property
    def _restricted_counts(
        self,
    ) -> Callable[[DemandRequest], DemandGraph | DemandReason | None]:
        return cache(partial(_count_restricted, self._intrinsic))

    def _restricted_result(
        self,
        request: DemandRequest | None,
        dense: NodeDict,
        profile: ArchitectureProfile,
    ) -> DemandResult | None:
        if request is None:
            return None
        counts = self._restricted_counts(request)
        if isinstance(counts, DemandReason):
            return DemandResult(
                graph_sha256=self._intrinsic.prepared.source_sha256,
                constraints=request,
                status=DemandStatus.BUDGET_EXHAUSTED
                if counts
                in {DemandReason.CELLS, DemandReason.TERMS, DemandReason.TIME}
                else DemandStatus.NOT_APPLICABLE,
                reason=counts,
                dense_macs=dense["total"]["macs"],
            )
        return demand_report(self._intrinsic, request, dense, counts, profile)

    def project(self, request: ProjectionRequest) -> ReuseResult:
        """Recompute attribution/capacity/pricing and publish one atomic result."""
        if analysis_rules_identity() != self._rules:
            raise ReuseRulesChangedError(
                "analysis rules changed; prepare a new reuse session"
            )
        output = request.output_dir.resolve()
        if output.exists():
            raise ReuseOutputExistsError(
                f"reuse output already exists: {output}"
            )
        configuration = configure_projection(
            request,
            self._program.architecture,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(
            prefix=f".{output.name}.", dir=output.parent
        ) as directory:
            staging = Path(directory)
            analysis = self._analyzer.project_intrinsic(
                self._intrinsic,
                output_dir=staging,
                architecture=configuration.profile,
                orojenesis_runner=configuration.runner,
                require_orojenesis=request.require_orojenesis,
                write_output=False,
            )
            restricted = self._restricted_result(
                request.restricted, analysis, configuration.profile
            )
            if restricted is not None:
                write_json_artifact(
                    staging / DEMAND_FILENAME,
                    restricted.model_dump(mode="json"),
                    max_bytes=request.output_bytes,
                )
            record = seal_projection(
                staging,
                analysis,
                self._program,
                request,
                configuration,
                self._intrinsic_identity,
                self._rules,
            )
            staging.replace(output)
        # Serialized bytes have passed the output budget. Return detached data
        # so a caller cannot mutate the session's graphs or identity records.
        return ReuseResult(
            output,
            deepcopy(analysis),
            record.model_copy(deep=True),
            restricted.model_copy(deep=True)
            if restricted is not None
            else None,
        )

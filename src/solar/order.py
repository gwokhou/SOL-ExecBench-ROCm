# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Budgeted contraction-order search with an unchanged original callable oracle."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from time import monotonic

from solar.analysis.exploration_contracts import ProjectionReference
from solar.analysis.identity import analysis_rules_identity
from solar.analysis.interrupts import (
    AnalysisDeadline,
    analysis_deadline,
    cleanup_on_termination,
)
from solar.analysis.order_contracts import (
    MAX_ORDER_EXECUTION_FIELDS,
    MAX_ORDER_EXECUTION_KEY_CHARS,
    MAX_ORDER_EXECUTION_VALUE_CHARS,
    MAX_ORDER_MESSAGE_CHARS,
    MAX_ORDER_REPORT_BYTES,
    ORDER_FILENAME,
    ORDER_PLAN_FILENAME,
    CandidateStatus,
    OrderCandidate,
    OrderResult,
    OrderStopReason,
)
from solar.analysis.order_domain import matrix_chain
from solar.analysis.order_selection import select_order
from solar.analysis.order_trees import (
    association_count,
    planned_trees,
    tree_cost,
    tree_identifier,
)
from solar.analysis.views import ViewAvailability
from solar.artifacts import sha256_file, stable_json_checksum
from solar.artifacts.json import write_json_artifact
from solar.artifacts.names import REUSE_FILENAME
from solar.contracts import ConversionRequest
from solar.errors import (
    ConversionVerificationError,
    GradientVerificationError,
    SolarError,
)
from solar.order_contracts import (
    OrderPreparationTimeoutError,
    OrderSearchRequest,
)
from solar.order_execution import (
    OrderContext,
    require_original_conversion,
    run_candidate,
)
from solar.pipeline.exploration import projection_views
from solar.pipeline.order_artifacts import resume_candidates, seal_candidate
from solar.pipeline.reuse_projection import configure_projection
from solar.pipeline.verified_program import load_verified_program
from solar.reuse import ReuseSession
from solar.reuse_contracts import ReuseResult, ReuseRulesChangedError
from solar.schema_versions import SOLAR_ORDER_SCHEMA_VERSION


def _projection_reference(
    result: ReuseResult, request: OrderSearchRequest
) -> ProjectionReference:
    return ProjectionReference(
        directory=result.output_dir,
        record_sha256=sha256_file(result.output_dir / REUSE_FILENAME),
        projection_sha256=result.record.projection_sha256,
        budget=request.job.source.budget,
    )


def search_contractions(
    conversion: ConversionRequest,
    request: OrderSearchRequest,
    *,
    execution_identity: dict[str, str] | None = None,
) -> OrderResult:
    """Search on CPU using the exact original reference, inputs and verification policy.

    ``conversion`` must be supplied by the caller; verified IR is never used as
    the numerical oracle. Use the isolated CLI worker for an outer hard timeout.
    """
    try:
        executor = _OrderExecutor(conversion, request, execution_identity or {})
    except AnalysisDeadline as exc:
        raise OrderPreparationTimeoutError(
            "source preparation exceeded the total search budget; no candidate plan was executed"
        ) from exc
    return executor.run()


class _OrderExecutor:
    def __init__(
        self,
        conversion: ConversionRequest,
        request: OrderSearchRequest,
        execution: dict[str, str],
    ) -> None:
        self.started = monotonic()
        self.request = request
        if len(execution) > MAX_ORDER_EXECUTION_FIELDS or any(
            not isinstance(key, str)
            or not isinstance(value, str)
            or len(key) > MAX_ORDER_EXECUTION_KEY_CHARS
            or len(value) > MAX_ORDER_EXECUTION_VALUE_CHARS
            for key, value in execution.items()
        ):
            raise ValueError(
                "search execution identity exceeds its field budget"
            )
        with analysis_deadline(request.budget.total_seconds):
            program = load_verified_program(request.job.source)
            require_original_conversion(conversion, program)
            chain = matrix_chain(program.graph.data)
            projection = request.job.projection.model_copy(
                update={"hypothetical": True}
            )
            configuration = configure_projection(
                projection,
                program.architecture,
            )
        self.context = OrderContext(
            request, conversion, program, chain, configuration
        )
        self.planned = tuple(
            OrderCandidate(
                candidate_id=tree_identifier(tree),
                tree=tree,
                cost=tree_cost(tree, chain.dimensions),
                original=tree == chain.original,
            )
            for tree in planned_trees(
                chain.original, chain.dimensions, request.budget.max_candidates
            )
        )
        self.identity = self._identity(execution)
        self.digest = stable_json_checksum(self.identity)
        self.output = projection.output_dir.resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self.candidates = list(self.planned)
        self.baseline: ReuseResult | None = None
        self.stop = OrderStopReason.INTERRUPTED
        write_json_artifact(
            self.output / ORDER_PLAN_FILENAME,
            {
                "schema_version": SOLAR_ORDER_SCHEMA_VERSION,
                "identity_inputs": self.identity,
                "candidates": [
                    item.model_dump(mode="json") for item in self.planned
                ],
            },
            max_bytes=MAX_ORDER_REPORT_BYTES,
        )

    def _identity(self, execution: dict[str, str]) -> dict[str, object]:
        # The serialized report budget also bounds caller-authored identity data.
        job = self.request.job.model_dump(mode="json")
        job["projection"].pop("output_dir")
        job["projection"]["architecture"] = (
            self.context.configuration.profile.to_dict()
        )
        job["projection"]["hypothetical"] = True
        return json.loads(
            json.dumps(
                {
                    "job": job,
                    "chain": asdict(self.context.chain),
                    "reference_name": self.context.program.manifest.reference.name,
                    "trace_seed": self.context.conversion.trace_seed,
                    "precision": self.context.program.manifest.analysis_contract.precision,
                    "view": self.request.view.value,
                    "computation": self.request.computation.value,
                    "budget": self.request.budget.model_dump(mode="json"),
                    "rules": analysis_rules_identity(),
                    "execution": execution,
                    "profile": self.context.configuration.profile.to_dict(),
                    "mapper": self.context.configuration.identity["mapper"],
                    "algebra": "ordered_pure_matrix_reassociation_subject_to_original_numerical_and_gradient_policy",
                    "tie_rule": "original_then_enumeration_order",
                    "concurrency": 1,
                },
                allow_nan=False,
            )
        )

    def _remaining(self) -> float:
        return self.request.budget.total_seconds - (monotonic() - self.started)

    def run(self) -> OrderResult:
        try:
            with cleanup_on_termination():
                self._run_prepared()
                if analysis_rules_identity() != self.identity["rules"]:
                    self.stop = OrderStopReason.IDENTITY_CHANGED
                    raise ReuseRulesChangedError(
                        "search rules changed during the attempt"
                    )
        finally:
            self.candidates = [
                item.model_copy(update={"reason": self.stop.value})
                if item.status is CandidateStatus.NOT_RUN
                else item
                for item in self.candidates
            ]
            result = self._result()
            write_json_artifact(
                self.output / ORDER_FILENAME,
                result.model_dump(mode="json"),
                max_bytes=MAX_ORDER_REPORT_BYTES,
            )
        return result

    def _run_prepared(self) -> None:
        if self._remaining() <= 0:
            self.stop = OrderStopReason.TOTAL_BUDGET
            return
        try:
            with analysis_deadline(self._remaining()):
                resumed = resume_candidates(
                    self.request, self.planned, self.identity
                )
                session = ReuseSession.from_verified_program(
                    self.context.program,
                    access_budget=self.request.job.access_budget,
                )
                self.baseline = session.project(
                    self.request.job.projection.model_copy(
                        update={
                            "architecture": self.context.configuration.profile.to_dict(),
                            "hypothetical": True,
                            "output_dir": self.output / "baseline",
                        }
                    )
                )
        except AnalysisDeadline:
            self.stop = OrderStopReason.TOTAL_BUDGET
            return
        for index, candidate in enumerate(self.planned):
            if candidate.candidate_id in resumed:
                self.candidates[index] = resumed[candidate.candidate_id]
                continue
            if self._remaining() <= 0:
                self.stop = OrderStopReason.TOTAL_BUDGET
                return
            self._attempt(index, candidate)
        self.stop = (
            OrderStopReason.COMPLETED
            if len(self.planned)
            == association_count(len(self.context.chain.source_indices))
            else OrderStopReason.CANDIDATE_BUDGET
        )

    def _attempt(self, index: int, candidate: OrderCandidate) -> None:
        started = monotonic()
        output = self.output / candidate.candidate_id
        output.mkdir()
        current = candidate.model_copy(
            update={
                "status": CandidateStatus.FAILED,
                "reason": "invariant_failure",
            }
        )
        try:
            if (
                candidate.cost.macs > self.request.budget.max_macs
                or candidate.cost.allocated_elements
                > self.request.budget.max_elements
            ):
                current = current.model_copy(
                    update={
                        "status": CandidateStatus.WORK_BUDGET,
                        "reason": "candidate_work_budget",
                    }
                )
            else:
                current = current.model_copy(update={"attempted": True})
                with analysis_deadline(
                    min(
                        self.request.budget.candidate_seconds,
                        max(self._remaining(), 1e-9),
                    )
                ):
                    current = self._evaluate(current, output)
        except AnalysisDeadline:
            current = current.model_copy(
                update={
                    "status": CandidateStatus.TIMEOUT,
                    "reason": "candidate_deadline",
                }
            )
        except (ConversionVerificationError, GradientVerificationError) as exc:
            current = current.model_copy(
                update={
                    "status": CandidateStatus.VERIFICATION_FAILED,
                    "reason": exc.reason_code,
                    "message": str(exc)[:MAX_ORDER_MESSAGE_CHARS],
                }
            )
        except SolarError as exc:
            current = current.model_copy(
                update={
                    "reason": exc.reason_code,
                    "message": str(exc)[:MAX_ORDER_MESSAGE_CHARS],
                }
            )
        except OSError as exc:
            current = current.model_copy(
                update={
                    "reason": "candidate_io_failure",
                    "message": str(exc)[:MAX_ORDER_MESSAGE_CHARS],
                }
            )
        finally:
            current = current.model_copy(
                update={"elapsed_seconds": monotonic() - started}
            )
            self.candidates[index] = seal_candidate(
                output, current, self.digest, self.request
            )

    def _evaluate(
        self, candidate: OrderCandidate, output: Path
    ) -> OrderCandidate:
        result = run_candidate(self.context, candidate, output)
        if result.record.rules != self.identity["rules"]:
            raise ValueError("search rules changed during candidate execution")
        views = projection_views(result, self.request.computation)
        return candidate.model_copy(
            update={
                "status": CandidateStatus.ANALYZED
                if views[self.request.view].status is ViewAvailability.AVAILABLE
                else CandidateStatus.UNAVAILABLE,
                "reason": views[self.request.view].reason.value,
                "views": views,
                "projection": _projection_reference(result, self.request),
            }
        )

    def _result(self) -> OrderResult:
        total = association_count(len(self.context.chain.source_indices))
        selection = select_order(
            self.candidates,
            self.request.view,
            total,
            completed=self.stop is OrderStopReason.COMPLETED,
        )
        return OrderResult(
            search_sha256=self.digest,
            identity_inputs=self.identity,
            chain=self.context.chain,
            baseline=_projection_reference(self.baseline, self.request)
            if self.baseline
            else None,
            baseline_views=projection_views(
                self.baseline, self.request.computation
            )
            if self.baseline
            else None,
            total_candidates=total,
            candidates=tuple(self.candidates),
            unexplored_candidates=selection.unexplored,
            selected_candidate_id=selection.candidate_id,
            selection_reason=selection.reason,
            exhaustive_verified_domain=selection.exhaustive_verified_domain,
            elapsed_seconds=monotonic() - self.started,
            stop_reason=self.stop,
            previous=self.request.resume,
        )

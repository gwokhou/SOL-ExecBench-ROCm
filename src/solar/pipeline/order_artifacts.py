# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Content-bound candidate inventories and exact-identity search recovery."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import cast

from solar.analysis.order_contracts import (
    MAX_ORDER_REPORT_BYTES,
    ORDER_CANDIDATE_FILENAME,
    ORDER_FILENAME,
    CandidateRecord,
    CandidateStatus,
    OrderCandidate,
    OrderReference,
    OrderResult,
    OrderStopReason,
)
from solar.analysis.order_domain import matrix_chain
from solar.analysis.order_selection import select_order
from solar.analysis.order_trees import (
    MAX_ORDER_CANDIDATES,
    association_count,
    tree_cost,
    tree_identifier,
)
from solar.analysis.views import AnalysisView, ViewAvailability
from solar.artifacts import sha256_bytes, sha256_file, stable_json_checksum
from solar.artifacts.bundle import (
    capture_bound_files,
    hash_bundle_files,
    read_regular_file,
)
from solar.artifacts.document import ArtifactMap, parse_yaml_artifact
from solar.artifacts.json import write_json_artifact
from solar.order_contracts import OrderSearchRequest
from solar.pipeline.exploration import projection_views, read_projection
from solar.pipeline.verified_program import load_verified_program
from solar.reuse_contracts import ProfileScenario
from solar.schema_versions import SOLAR_ORDER_SCHEMA_VERSION


def _read_current(path: Path, digest: str) -> ArtifactMap:
    payload = read_regular_file(path, max_bytes=MAX_ORDER_REPORT_BYTES)
    if sha256_bytes(payload) != digest:
        raise ValueError("search artifact checksum differs from pin")
    document = parse_yaml_artifact(
        path, payload, max_bytes=MAX_ORDER_REPORT_BYTES
    )
    if (
        type(document.get("schema_version")) is not int
        or document.get("schema_version") != SOLAR_ORDER_SCHEMA_VERSION
    ):
        raise ValueError("contraction search schema is not current")
    return document.data


def load_order_result(reference: OrderReference) -> OrderResult:
    """Validate the finite report and tree-derived identities at the read boundary."""
    raw = _read_current(
        reference.directory / ORDER_FILENAME, reference.record_sha256
    )
    candidates = raw.get("candidates")
    if (
        not isinstance(candidates, list)
        or not 1 <= len(candidates) <= MAX_ORDER_CANDIDATES
    ):
        raise ValueError("search report exceeds the candidate budget")
    result = OrderResult.model_validate(raw)
    if stable_json_checksum(result.identity_inputs) != result.search_sha256:
        raise ValueError("search identity checksum differs")
    if len({item.candidate_id for item in result.candidates}) != len(
        result.candidates
    ):
        raise ValueError("search contains duplicate candidates")
    for item in result.candidates:
        if item.candidate_id != tree_identifier(
            item.tree
        ) or item.cost != tree_cost(item.tree, result.chain.dimensions):
            raise ValueError("candidate tree, identifier or cost differs")
        if item.original != (item.tree == result.chain.original):
            raise ValueError("candidate original flag differs")
    if not result.candidates[0].original:
        raise ValueError("search must begin with the original association")
    total = association_count(len(result.chain.source_indices))
    view = AnalysisView(result.identity_inputs["view"])
    selected = select_order(
        result.candidates,
        view,
        total,
        completed=result.stop_reason is OrderStopReason.COMPLETED,
    )
    if (
        result.total_candidates != total
        or result.selected_candidate_id != selected.candidate_id
        or result.selection_reason != selected.reason
        or result.exhaustive_verified_domain
        != selected.exhaustive_verified_domain
        or result.unexplored_candidates != selected.unexplored
    ):
        raise ValueError(
            "search selection or coverage differs from its candidate records"
        )
    _bind_result(reference, result)
    return result


def _bind_result(reference: OrderReference, result: OrderResult) -> None:
    job = result.identity_inputs["job"]
    if not isinstance(job, dict) or not isinstance(job.get("projection"), dict):
        raise ValueError("search lacks its source/projection identity")
    job = cast(dict[str, object], job)
    projection = cast(dict[str, object], job["projection"])
    request = OrderSearchRequest.model_validate(
        {
            "job": {
                **job,
                "projection": {
                    **projection,
                    "output_dir": reference.directory,
                },
            },
            "view": result.identity_inputs["view"],
            "computation": result.identity_inputs["computation"],
            "budget": result.identity_inputs["budget"],
        }
    )
    for candidate in result.candidates:
        if candidate.status is not CandidateStatus.NOT_RUN:
            _accept_candidate(candidate, request, result.identity_inputs)
    if result.baseline is not None:
        baseline = read_projection(result.baseline)
        if (
            result.baseline.directory.resolve()
            != reference.directory.resolve() / "baseline"
            or baseline.record.source_manifest_sha256
            != request.job.source.manifest_sha256
            or baseline.record.architecture_sha256
            != stable_json_checksum(result.identity_inputs["profile"])
            or baseline.record.rules != result.identity_inputs["rules"]
            or baseline.record.scenario is not ProfileScenario.HYPOTHETICAL
            or baseline.record.sol_score_eligible
            or baseline.record.publication_eligible
            or projection_views(baseline, request.computation)
            != result.baseline_views
        ):
            raise ValueError(
                "search baseline differs from its pinned source or model"
            )
    elif result.baseline_views is not None:
        raise ValueError("search baseline views have no bound projection")


def seal_candidate(
    output: Path,
    candidate: OrderCandidate,
    identity: str,
    request: OrderSearchRequest,
) -> OrderCandidate:
    """Bind all retained files, including failed extraction/verification work."""
    record = CandidateRecord(
        search_sha256=identity,
        candidate=candidate,
        artifacts=hash_bundle_files(output, budget=request.job.source.budget),
    )
    path = output / ORDER_CANDIDATE_FILENAME
    write_json_artifact(
        path, record.model_dump(mode="json"), max_bytes=MAX_ORDER_REPORT_BYTES
    )
    return candidate.model_copy(
        update={
            "bundle": OrderReference(
                directory=output, record_sha256=sha256_file(path)
            )
        }
    )


def _accept_candidate(
    candidate: OrderCandidate,
    request: OrderSearchRequest,
    identity: dict[str, object],
) -> None:
    if candidate.bundle is None:
        raise ValueError(
            "completed candidate lacks its complete artifact inventory"
        )
    reference = candidate.bundle
    raw = _read_current(
        reference.directory / ORDER_CANDIDATE_FILENAME, reference.record_sha256
    )
    artifacts = raw.get("artifacts")
    if (
        not isinstance(artifacts, dict)
        or len(artifacts) > request.job.source.budget.files
    ):
        raise ValueError("candidate artifact inventory exceeds budget")
    record = CandidateRecord.model_validate(raw)
    expected = candidate.model_copy(update={"bundle": None, "resumed": False})
    if (
        record.search_sha256 != stable_json_checksum(identity)
        or record.candidate != expected
    ):
        raise ValueError("candidate record differs from search report")
    capture_bound_files(
        reference.directory,
        record.artifacts,
        retain=frozenset(),
        budget=request.job.source.budget,
        manifest_name=ORDER_CANDIDATE_FILENAME,
    )
    if candidate.projection is not None:
        _accept_projection(candidate, request, identity)


def _accept_projection(
    candidate: OrderCandidate,
    request: OrderSearchRequest,
    identity: dict[str, object],
) -> None:
    if candidate.projection is None or candidate.bundle is None:
        raise ValueError("completed candidate lacks projection or bundle")
    if (
        candidate.projection.directory.resolve()
        != candidate.bundle.directory.resolve() / "projection"
    ):
        raise ValueError("candidate projection is outside its bound bundle")
    projected = read_projection(candidate.projection)
    record = projected.record
    projection = record.identity_inputs["projection"]
    if (
        record.source_reference_sha256 != request.job.source.reference_sha256
        or record.source_ir_path is not request.job.source.ir_path
        or record.scenario is not ProfileScenario.HYPOTHETICAL
        or record.rules != identity["rules"]
        or record.sol_score_eligible
        or record.publication_eligible
        or record.profile_audit_verified
        or not isinstance(projection, dict)
        or projection.get("restricted")
        != (
            request.job.projection.restricted.model_dump(mode="json")
            if request.job.projection.restricted
            else None
        )
        or projection.get("architecture") != identity["profile"]
        or projection.get("mapper") != identity["mapper"]
    ):
        raise ValueError(
            "candidate projection source/path/profile/model/qualification differs"
        )
    views = projection_views(projected, request.computation)
    available = views[request.view].status is ViewAvailability.AVAILABLE
    if (
        candidate.views != views
        or (candidate.status is CandidateStatus.ANALYZED) != available
    ):
        raise ValueError("candidate views differ from their bound analysis")
    _accept_candidate_source(
        candidate, request, identity, record.source_manifest_sha256
    )


def _accept_candidate_source(
    candidate: OrderCandidate,
    request: OrderSearchRequest,
    identity: dict[str, object],
    manifest_sha256: str,
) -> None:
    if candidate.bundle is None:
        raise ValueError("candidate has no source bundle")
    source = request.job.source.model_copy(
        update={
            "directory": candidate.bundle.directory / "full",
            "manifest_sha256": manifest_sha256,
        }
    )
    program = load_verified_program(source)
    chain = matrix_chain(program.graph.data)
    original = identity["chain"]
    if not isinstance(original, dict):
        raise ValueError("search identity lacks its source chain")
    original = cast(dict[str, object], original)
    if (
        chain.original != candidate.tree
        or list(chain.dimensions) != original["dimensions"]
        or list(chain.source_indices) != original["source_indices"]
        or chain.dtype != original["dtype"]
        or program.manifest.reference.name != identity["reference_name"]
        or program.manifest.analysis_contract.trace_seed
        != identity["trace_seed"]
        or program.manifest.analysis_contract.precision != identity["precision"]
    ):
        raise ValueError(
            "candidate executable source differs from the planned association"
        )


def resume_candidates(
    request: OrderSearchRequest,
    planned: tuple[OrderCandidate, ...],
    identity: dict[str, object],
) -> dict[str, OrderCandidate]:
    """Reuse terminal results only; all prior failures remain in the old attempt."""
    if request.resume is None:
        return {}
    previous = load_order_result(request.resume)
    if previous.search_sha256 != stable_json_checksum(identity) or len(
        previous.candidates
    ) != len(planned):
        raise ValueError("resumed contraction search identity differs")
    if stable_json_checksum(asdict(previous.chain)) != stable_json_checksum(
        identity["chain"]
    ):
        raise ValueError("resumed source chain differs")
    accepted = {}
    for prior, current in zip(previous.candidates, planned, strict=True):
        if (
            prior.tree != current.tree
            or prior.cost != current.cost
            or prior.candidate_id != current.candidate_id
        ):
            raise ValueError("resumed candidate plan differs")
        if prior.status is CandidateStatus.NOT_RUN:
            continue
        # The loader checked each complete bundle; only carry it across an identical plan.
        if prior.status in {
            CandidateStatus.ANALYZED,
            CandidateStatus.UNAVAILABLE,
            CandidateStatus.VERIFICATION_FAILED,
            CandidateStatus.WORK_BUDGET,
        }:
            accepted[prior.candidate_id] = prior.model_copy(
                update={"resumed": True}
            )
    return accepted

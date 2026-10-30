# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Extract a new association, verify against the original callable, then price it."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from solar.analysis.order_contracts import OrderCandidate
from solar.analysis.order_domain import MatrixChain, matrix_chain
from solar.analysis.order_trees import MATMUL_TOKEN, OrderTree
from solar.artifacts import sha256_file, stable_json_checksum
from solar.artifacts.names import (
    CONVERSION_ATTESTATION_FILENAME,
    MANIFEST_FILENAME,
)
from solar.contracts import (
    FORMAL_BOUND_KIND,
    AnalysisRequest,
    ConversionRequest,
    write_request_manifest,
)
from solar.graph.extraction import extract_operator_graph
from solar.order_contracts import OrderSearchRequest
from solar.pipeline.publication import (
    extract_bound,
    finish_artifacts,
    verify_publication_tree,
)
from solar.pipeline.reuse_projection import ProjectionConfiguration
from solar.pipeline.stages import (
    analyze_request_graph,
    convert_request_graph,
    verify_request_graph,
)
from solar.pipeline.verified_program import (
    VerifiedProgram,
    load_verified_program,
)
from solar.reuse import ReuseSession
from solar.reuse_contracts import ReuseResult, VerifiedProgramRequest
from solar.types import DynamicValue
from solar.verification.policy import verification_policy_record


@dataclass(frozen=True)
class OrderContext:
    """One owned source, immutable domain and common pricing configuration."""

    request: OrderSearchRequest
    conversion: ConversionRequest
    program: VerifiedProgram
    chain: MatrixChain
    configuration: ProjectionConfiguration


def require_original_conversion(
    conversion: ConversionRequest, program: VerifiedProgram
) -> None:
    """Bind the supplied oracle to the exact original policy and source contract."""
    manifest, identity = program.manifest, program.identity
    if (
        conversion.reference_sha256 != identity.reference_sha256
        or conversion.reference_name != manifest.reference.name
        or conversion.ir_path is not identity.ir_path
        or conversion.trace_seed != manifest.analysis_contract.trace_seed
        or tuple(conversion.verification.seeds)
        != tuple(identity.verification.seeds)
        or tuple(conversion.verification.patterns)
        != tuple(identity.verification.patterns)
        or conversion.verification.device != identity.verification.device
        or verification_policy_record(conversion.verification)
        != verification_policy_record(identity.verification)
    ):
        raise ValueError(
            "search oracle source, path, trace seed or verification policy differs"
        )


def candidate_callable(
    chain: MatrixChain, tree: OrderTree
) -> Callable[..., DynamicValue]:
    """Build a pure tensor program from the already planned association."""

    def evaluate(*inputs: DynamicValue) -> DynamicValue:
        stack: list[DynamicValue] = []
        for token in tree:
            if token == MATMUL_TOKEN:
                right, left = stack.pop(), stack.pop()
                stack.append(left @ right)
            else:
                stack.append(inputs[chain.source_indices[token]])
        return stack[0]

    return evaluate


def run_candidate(
    context: OrderContext, candidate: OrderCandidate, output: Path
) -> ReuseResult:
    """Retain intermediate files on failure; only verified graphs get a manifest."""
    full = output / "full"
    full.mkdir()
    conversion = replace(context.conversion, analysis_id=candidate.candidate_id)
    request = AnalysisRequest(
        conversion=conversion,
        architecture=context.configuration.profile.to_dict(),
        output_dir=full,
        precision=context.program.manifest.analysis_contract.precision,
        require_verified_audit=False,
    )
    # The candidate is used ONLY for extraction. Verification below receives
    # the original callable and its unchanged complete policy.
    operator = extract_operator_graph(
        candidate_callable(context.chain, candidate.tree),
        tuple(conversion.input_factory(conversion.trace_seed)),
        device=request.device,
        output_dir=full,
        name=candidate.candidate_id,
        extraction_kind=request.extraction_kind,
    )
    graph = convert_request_graph(request, operator, full)
    observed = matrix_chain(graph.document.data)
    if (
        observed.original != candidate.tree
        or observed.dimensions != context.chain.dimensions
        or observed.source_indices != context.chain.source_indices
        or observed.dtype != context.chain.dtype
    ):
        raise ValueError(
            "extracted candidate differs from its planned association"
        )
    verify_request_graph(request, graph, full / CONVERSION_ATTESTATION_FILENAME)
    analysis = analyze_request_graph(
        request, context.configuration.profile, full, graph
    )
    if analysis["total"]["macs"] != candidate.cost.macs:
        raise ValueError(
            "candidate accounting differs from exact association MACs"
        )
    bound = extract_bound(analysis, False)
    artifacts = finish_artifacts(full, analysis, graph.path.name)
    write_request_manifest(
        request,
        full,
        stable_json_checksum(context.configuration.profile.to_dict()),
        artifacts,
        bound,
        formal_bound_kind=FORMAL_BOUND_KIND,
    )
    verify_publication_tree(full, artifacts)
    source = VerifiedProgramRequest(
        directory=full,
        manifest_sha256=sha256_file(full / MANIFEST_FILENAME),
        reference_sha256=conversion.reference_sha256,
        execution_sha256=context.request.job.source.execution_sha256,
        ir_path=conversion.ir_path,
        verification=conversion.verification,
        budget=context.request.job.source.budget,
    )
    session = ReuseSession.from_verified_program(
        load_verified_program(source),
        access_budget=context.request.job.access_budget,
    )
    projection = context.request.job.projection.model_copy(
        update={
            "architecture": context.configuration.profile.to_dict(),
            "hypothetical": True,
            "output_dir": output / "projection",
        }
    )
    return session.project(projection)

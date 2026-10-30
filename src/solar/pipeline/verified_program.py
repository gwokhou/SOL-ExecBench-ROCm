# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Load a pinned, complete analysis bundle without executing its reference."""

from __future__ import annotations

from dataclasses import dataclass

from solar.analysis.graph_validation import validate_current_ir_schema
from solar.artifacts import ArtifactDocument, sha256_bytes, stable_json_checksum
from solar.artifacts.bundle import (
    capture_bound_files,
    read_regular_file,
)
from solar.artifacts.document import ArtifactMap, parse_yaml_artifact
from solar.artifacts.names import (
    ANALYSIS_FILENAME,
    CONVERSION_ATTESTATION_FILENAME,
    MANIFEST_FILENAME,
)
from solar.contracts import (
    FORMAL_BOUND_KIND,
    SOL_BOUND_KINDS,
    SolarRequestManifest,
)
from solar.graph.contracts import OPERATOR_GRAPH_FILENAME
from solar.ir.access_artifact import (
    ACCESS_FILENAME,
    AccessDocument,
    parse_access_document,
)
from solar.ir.registry import ir_backend
from solar.reuse_contracts import VerifiedProgramRequest
from solar.schema_versions import (
    OPERATOR_GRAPH_SCHEMA_VERSION,
    SOLAR_REQUEST_MANIFEST_SCHEMA_VERSION,
)
from solar.verification.policy import verification_cases
from solar.verification.receipt import validate_callable_receipt

MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_REUSE_LAYERS = 10000


@dataclass(frozen=True)
class VerifiedProgram:
    """Owned, hash-checked data; no executable reference or pickle is loaded."""

    identity: VerifiedProgramRequest
    manifest: SolarRequestManifest
    graph: ArtifactDocument
    access: AccessDocument
    prior_analysis: ArtifactDocument

    @property
    def architecture(self) -> ArtifactMap:
        """Read the source profile from the captured, manifest-bound analysis."""
        metadata = self.prior_analysis.require_mapping("metadata")
        return ArtifactDocument(
            self.prior_analysis.path, metadata
        ).require_mapping("architecture")


def _manifest(request: VerifiedProgramRequest) -> SolarRequestManifest:
    path = request.directory / MANIFEST_FILENAME
    payload = read_regular_file(path, max_bytes=MAX_MANIFEST_BYTES)
    if sha256_bytes(payload) != request.manifest_sha256:
        raise ValueError(
            "request manifest SHA-256 differs from pinned identity"
        )
    raw = parse_yaml_artifact(path, payload, max_bytes=MAX_MANIFEST_BYTES).data
    if raw.get("schema_version") != SOLAR_REQUEST_MANIFEST_SCHEMA_VERSION:
        raise ValueError("request manifest schema is not current")
    artifacts = raw.get("artifacts")
    if isinstance(artifacts, list) and len(artifacts) > request.budget.files:
        raise ValueError("bundle exceeds the file count budget")
    manifest = SolarRequestManifest.model_validate(raw)
    contract = manifest.analysis_contract
    if (
        manifest.reference.sha256 != request.reference_sha256
        or contract.ir_path != request.ir_path
        or contract.ir_kind != request.ir_path.ir_kind
        or contract.extraction_kind != request.ir_path.extraction_kind
    ):
        raise ValueError("request source or IR path identity differs")
    if (
        contract.verification_seeds != list(request.verification.seeds)
        or any(
            getattr(contract, name) != getattr(request.verification, name)
            for name in (
                "atol",
                "rtol",
                "required_matched_ratio",
                "max_error_cap",
                "allow_negative_inf",
            )
        )
        or contract.preserved_input_indices
        != list(request.verification.preserved_input_indices)
    ):
        raise ValueError("request manifest verification policy differs")
    return manifest


def load_verified_program(request: VerifiedProgramRequest) -> VerifiedProgram:
    """Capture the entire manifest inventory before validating semantic claims."""
    cases = verification_cases({}, request.verification)
    manifest = _manifest(request)
    digests = {item.path: item.sha256 for item in manifest.artifacts}
    if len(digests) != len(manifest.artifacts):
        raise ValueError("request manifest contains duplicate artifact paths")
    graph_name = request.ir_path.graph_filename
    required = frozenset(
        {
            graph_name,
            OPERATOR_GRAPH_FILENAME,
            ACCESS_FILENAME,
            CONVERSION_ATTESTATION_FILENAME,
            ANALYSIS_FILENAME,
        }
    )
    captured = capture_bound_files(
        request.directory, digests, retain=required, budget=request.budget
    )
    graph, access = _validated_graph(request, captured, digests)
    receipt = parse_yaml_artifact(
        request.directory / CONVERSION_ATTESTATION_FILENAME,
        captured[CONVERSION_ATTESTATION_FILENAME],
    )
    validate_callable_receipt(
        receipt,
        subjects={
            manifest.reference.name: request.reference_sha256,
            graph_name: digests[graph_name],
            ACCESS_FILENAME: digests[ACCESS_FILENAME],
        },
        policy=request.verification,
        cases=cases,
        execution_sha256=request.execution_sha256,
    )
    analysis = parse_yaml_artifact(
        request.directory / ANALYSIS_FILENAME, captured[ANALYSIS_FILENAME]
    )
    _require_analysis_binding(analysis, manifest, digests[graph_name])
    return VerifiedProgram(request, manifest, graph, access, analysis)


def _validated_graph(
    request: VerifiedProgramRequest,
    captured: dict[str, bytes],
    digests: dict[str, str],
) -> tuple[ArtifactDocument, AccessDocument]:
    operator = parse_yaml_artifact(
        request.directory / OPERATOR_GRAPH_FILENAME,
        captured[OPERATOR_GRAPH_FILENAME],
    )
    if operator.get("schema_version") != OPERATOR_GRAPH_SCHEMA_VERSION:
        raise ValueError("operator graph schema is not current")
    if operator.get("extraction_kind") != request.ir_path.extraction_kind:
        raise ValueError(
            "operator extraction path differs from requested identity"
        )
    graph_name = request.ir_path.graph_filename
    graph = parse_yaml_artifact(
        request.directory / graph_name, captured[graph_name]
    )
    validate_current_ir_schema(graph)
    layers = graph.require_mapping("layers")
    if len(layers) > MAX_REUSE_LAYERS:
        raise ValueError("verified graph exceeds the layer budget")
    ir_backend(request.ir_path.ir_kind).validate(graph)
    access = parse_access_document(captured[ACCESS_FILENAME])
    access.require_digests(
        digests[graph_name], digests[OPERATOR_GRAPH_FILENAME]
    )
    access.require_graph(graph)
    return graph, access


def _require_analysis_binding(
    analysis: ArtifactDocument,
    manifest: SolarRequestManifest,
    graph_sha256: str,
) -> None:
    from solar.pipeline.publication import extract_bound

    bound = extract_bound(
        analysis, manifest.analysis_contract.require_orojenesis
    )
    metadata = analysis.require_mapping("metadata")
    if (
        metadata.get("source_graph_sha256") != graph_sha256
        or metadata.get("precision") != manifest.analysis_contract.precision
        or stable_json_checksum(metadata.get("architecture"))
        != manifest.architecture_sha256
        or bound.seconds != manifest.bound.seconds
        or bound.kind != manifest.bound.kind
        or bound.limiting_resource != manifest.bound.limiting_resource
        or manifest.sol_score_eligible != (bound.kind in SOL_BOUND_KINDS)
        or manifest.publication_eligible != (bound.kind == FORMAL_BOUND_KIND)
    ):
        raise ValueError("analysis identity or bound differs from its manifest")

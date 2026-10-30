# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOLAR ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Seal and read complete derived results with their exact reuse identity."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from solar.analysis.demand_contracts import (
    DEMAND_FILENAME,
    DemandRequest,
    DemandResult,
)
from solar.analysis.identity import RuleGroup
from solar.artifacts import sha256_bytes, stable_json_checksum
from solar.artifacts.bundle import (
    DEFAULT_BUNDLE_BUDGET,
    BundleBudget,
    capture_bound_files,
    hash_bundle_files,
    read_regular_file,
)
from solar.artifacts.document import (
    ArtifactDocument,
    ArtifactMap,
    parse_yaml_artifact,
)
from solar.artifacts.json import write_json_artifact
from solar.artifacts.names import ANALYSIS_FILENAME, REUSE_FILENAME
from solar.pipeline.publication import extract_bound
from solar.pipeline.reuse_projection import ProjectionConfiguration
from solar.pipeline.verified_program import VerifiedProgram
from solar.reuse_contracts import (
    ProfileScenario,
    ProjectionRequest,
    ReuseRecord,
    ReuseResult,
)
from solar.schema_versions import (
    SOLAR_DEMAND_SCHEMA_VERSION,
    SOLAR_REUSE_SCHEMA_VERSION,
)


def seal_projection(
    staging: Path,
    analysis: ArtifactMap,
    program: VerifiedProgram,
    request: ProjectionRequest,
    configuration: ProjectionConfiguration,
    intrinsic_identity: dict[str, object],
    rules: dict[RuleGroup, str],
) -> ReuseRecord:
    """Write a derived analysis and bind all emitted capacity artifacts."""
    bound = extract_bound(analysis, request.require_orojenesis)
    cast(ArtifactMap, analysis["metadata"])["source_graph"] = (
        program.identity.ir_path.graph_filename
    )
    write_json_artifact(
        staging / ANALYSIS_FILENAME, analysis, max_bytes=request.output_bytes
    )
    artifacts = hash_bundle_files(staging, budget=program.identity.budget)
    inputs = {
        "intrinsic": intrinsic_identity,
        "projection": configuration.identity,
        "rules": rules,
    }
    unchanged = (
        configuration.scenario is ProfileScenario.IDENTICAL
        and request.restricted is None
    )
    same_kind = bound.kind == program.manifest.bound.kind
    record = ReuseRecord(
        source_manifest_sha256=program.identity.manifest_sha256,
        source_reference_sha256=program.identity.reference_sha256,
        source_ir_path=program.identity.ir_path,
        intrinsic_sha256=stable_json_checksum(intrinsic_identity),
        projection_sha256=stable_json_checksum(inputs),
        analysis_sha256=artifacts[ANALYSIS_FILENAME],
        artifacts=artifacts,
        requested_architecture_sha256=configuration.requested_sha256,
        architecture_sha256=stable_json_checksum(
            configuration.profile.to_dict()
        ),
        rules=rules,
        scenario=configuration.scenario,
        profile_audit_verified=configuration.audit_verified,
        sol_score_eligible=unchanged
        and program.manifest.sol_score_eligible
        and same_kind,
        publication_eligible=unchanged
        and configuration.audit_verified
        and program.manifest.publication_eligible
        and same_kind,
        identity_inputs=inputs,
    )
    write_json_artifact(
        staging / REUSE_FILENAME,
        record.model_dump(mode="json"),
        max_bytes=request.output_bytes,
    )
    return record


def load_projection(
    directory: Path,
    *,
    record_sha256: str,
    projection_sha256: str,
    budget: BundleBudget = DEFAULT_BUNDLE_BUDGET,
) -> ReuseResult:
    """Read only a pinned, fully emitted result; never treat partial output as done."""
    path = directory / REUSE_FILENAME
    payload = read_regular_file(path, max_bytes=budget.file_bytes)
    if sha256_bytes(payload) != record_sha256:
        raise ValueError("reuse record SHA-256 differs from pinned identity")
    document = parse_yaml_artifact(path, payload)
    if (
        type(document.get("schema_version")) is not int
        or document.get("schema_version") != SOLAR_REUSE_SCHEMA_VERSION
    ):
        raise ValueError("reuse record schema is not current")
    record = ReuseRecord.model_validate(document.data)
    if (
        record.projection_sha256 != projection_sha256
        or stable_json_checksum(record.identity_inputs) != projection_sha256
        or stable_json_checksum(record.identity_inputs.get("intrinsic"))
        != record.intrinsic_sha256
        or record.artifacts.get(ANALYSIS_FILENAME) != record.analysis_sha256
    ):
        raise ValueError("reuse projection identity differs")
    captured = capture_bound_files(
        directory,
        record.artifacts,
        retain=frozenset(
            {ANALYSIS_FILENAME} | ({DEMAND_FILENAME} & set(record.artifacts))
        ),
        budget=budget,
        manifest_name=REUSE_FILENAME,
    )
    analysis = parse_yaml_artifact(
        directory / ANALYSIS_FILENAME, captured[ANALYSIS_FILENAME]
    )
    extract_bound(analysis, False)
    if (
        stable_json_checksum(
            analysis.require_mapping("metadata").get("architecture")
        )
        != record.architecture_sha256
    ):
        raise ValueError("reuse analysis architecture identity differs")
    restricted = _read_restricted(directory, captured, record, analysis)
    return ReuseResult(directory, analysis.data, record, restricted)


def _read_restricted(
    directory: Path,
    captured: dict[str, bytes],
    record: ReuseRecord,
    analysis: ArtifactDocument,
) -> DemandResult | None:
    projection = record.identity_inputs.get("projection")
    if not isinstance(projection, dict):
        raise ValueError("projection identity is malformed")
    raw = projection.get("restricted")
    if raw is None:
        if DEMAND_FILENAME in record.artifacts:
            raise ValueError("unexpected restricted computation artifact")
        return None
    constraints = DemandRequest.model_validate(raw)
    if DEMAND_FILENAME not in captured:
        raise ValueError("restricted computation artifact is missing")
    document = parse_yaml_artifact(
        directory / DEMAND_FILENAME, captured[DEMAND_FILENAME]
    )
    if (
        type(document.get("schema_version")) is not int
        or document.get("schema_version") != SOLAR_DEMAND_SCHEMA_VERSION
    ):
        raise ValueError("restricted computation schema is not current")
    result = DemandResult.model_validate(document.data)
    if (
        result.constraints != constraints
        or result.graph_sha256
        != analysis.require_mapping("metadata")["source_graph_sha256"]
        or result.dense_macs != analysis.require_mapping("total")["macs"]
    ):
        raise ValueError(
            "restricted computation source/constraints/counts differ"
        )
    return result

"""Authenticate worker evidence before using it in a research coverage row."""

from pathlib import Path

from sol_execbench.core.dataset.target_view import (
    MaterializedTargetView,
    scoped_artifact,
)
from sol_execbench.core.integrity import sha256_bytes, sha256_file
from sol_execbench.core.solar_bridge.corpus_readiness import (
    verify_readiness_receipt,
)
from sol_execbench.core.solar_bridge.coverage_models import (
    CoverageRow,
    CoverageStatus,
)
from sol_execbench.core.solar_bridge.models import (
    SolarAnalysisOutcome,
    SolarStageAuditOutcome,
)
from sol_execbench.core.solar_bridge.performance import (
    load_manifest_semantic_characterization,
)
from sol_execbench.core.solar_bridge.publication import (
    verified_solar_artifact_paths,
)
from sol_execbench.core.solar_bridge.workload_context import (
    workload_verification_policy,
)
from solar.artifacts import load_yaml_artifact
from solar.artifacts.names import (
    ANALYSIS_FILENAME,
    CONVERSION_ATTESTATION_FILENAME,
    MANIFEST_FILENAME,
)
from solar.contracts import SolarRequestManifest
from solar.errors import MAPPER_TIMEOUT_REASON_CODE

_BUDGET_REASONS = frozenset(
    {
        "worker_timeout",
        "orojenesis_timeout",
        MAPPER_TIMEOUT_REASON_CODE,
        "memory_budget_exhausted",
        "evidence_budget_exhausted",
    }
)
_UNSUPPORTED_REASONS = frozenset(
    {"unsupported_operation", "unsupported_dtype", "unsupported_ir_operation"}
)
_UNPROVED_REASONS = frozenset(
    {
        "conversion_not_proven",
        "external_reads_unproved",
        "formal_bound_unproved",
    }
)


def classify_failure(reason: str | None, stage: str | None) -> CoverageStatus:
    """Never infer unsupported from an unknown reason or a generic crash."""
    if reason in _BUDGET_REASONS:
        return CoverageStatus.BUDGET_EXHAUSTED
    if reason in _UNSUPPORTED_REASONS:
        return CoverageStatus.UNSUPPORTED
    if reason in _UNPROVED_REASONS:
        return CoverageStatus.UNPROVED
    if stage in {None, "outer_bridge", "prepare", "architecture"} or (
        reason or ""
    ).startswith("worker_"):
        return CoverageStatus.INFRASTRUCTURE_ERROR
    return CoverageStatus.FAILED


def verify_stage_outcome(
    target: MaterializedTargetView,
    row: CoverageRow,
    outcome: SolarStageAuditOutcome,
    directory: Path,
    device: str,
) -> dict[str, str]:
    """Check exact path, workload, hashes and current semantic receipt."""
    definition = target.definitions[row.semantic_id]
    admitted_ids = {f"{definition.name}:{row.workload_uuid}"}
    if not outcome.ready:
        admitted_ids.add(row.workload_uuid)
    if (
        outcome.ir_path is not row.ir_path
        or outcome.analysis_id not in admitted_ids
    ):
        raise ValueError("readiness response identity mismatch")
    hashes = _verify_members(outcome.artifacts, outcome.output_dir, directory)
    if outcome.ready:
        verify_readiness_receipt(
            {
                "stages": list(outcome.stages),
                "reference_sha256": sha256_bytes(definition.reference.encode()),
            },
            directory.resolve(),
            workload_verification_policy(
                definition, target.workloads[row.workload_uuid], device
            ),
            f"{definition.name}/definition.json#reference",
        )
    return hashes


def verify_formal_outcome(
    target: MaterializedTargetView,
    row: CoverageRow,
    outcome: SolarAnalysisOutcome,
    directory: Path,
) -> tuple[dict[str, str], dict[str, object]]:
    """Use production manifest readers, preserving raw counts and view assumptions."""
    if outcome.ir_path is not row.ir_path:
        raise ValueError("formal response IR path mismatch")
    hashes = _verify_members(outcome.artifacts, outcome.output_dir, directory)
    if not outcome.is_formal_publication:
        return hashes, {}
    path = scoped_artifact(directory, MANIFEST_FILENAME)
    manifest = SolarRequestManifest.from_yaml(path.read_text())
    if (
        manifest.analysis_contract.ir_path != row.ir_path
        or not manifest.publication_eligible
    ):
        raise ValueError("formal manifest path or eligibility mismatch")
    for member in verified_solar_artifact_paths(path):
        hashes[str(member.relative_to(directory))] = sha256_file(member)
    hashes[MANIFEST_FILENAME] = sha256_file(path)
    characterization = load_manifest_semantic_characterization(
        path,
        workload_uuid=row.workload_uuid,
        definition=target.definitions[row.semantic_id].name,
    )
    analysis = load_yaml_artifact(directory / ANALYSIS_FILENAME).data
    return hashes, {
        "characterization": characterization.model_dump(mode="json"),
        "total": analysis.get("total"),
        "metadata": analysis.get("metadata"),
    }


def _verify_members(
    members: tuple[dict[str, str], ...],
    reported_directory: str | None,
    expected: Path,
) -> dict[str, str]:
    if (
        reported_directory is not None
        and Path(reported_directory).resolve() != expected.resolve()
    ):
        raise ValueError("worker output directory mismatch")
    hashes: dict[str, str] = {}
    for member in members:
        relative = member["path"]
        if relative in hashes:
            raise ValueError("duplicate worker artifact")
        path = scoped_artifact(expected, relative)
        if sha256_file(path) != member["sha256"]:
            raise ValueError("worker artifact checksum mismatch")
        hashes[relative] = member["sha256"]
    return hashes


def verification_dimensions(directory: Path) -> dict[str, object]:
    """Project only explicit receipt fields after the caller authenticates it."""
    predicate = load_yaml_artifact(
        directory / CONVERSION_ATTESTATION_FILENAME
    ).require_mapping("predicate")
    results = predicate.get("results")
    if not isinstance(results, list):
        raise ValueError("verification receipt has no result cases")
    gradients = []
    for result in results:
        if not isinstance(result, dict):
            raise ValueError("verification case is not a mapping")
        count = result.get("gradient_inputs_verified")
        error_semantics = result.get("gradient_error_semantics_verified")
        status = CoverageStatus.NOT_OBSERVABLE
        if isinstance(count, (float, int)):
            status = (
                CoverageStatus.PASSED
                if count > 0
                else CoverageStatus.NOT_APPLICABLE
            )
        if error_semantics == 1:
            status = CoverageStatus.PASSED
        gradients.append(
            {
                "status": status,
                "inputs_verified": count,
                "error_semantics_verified": error_semantics,
            }
        )
    return {
        "numerical": {"status": CoverageStatus.PASSED, "cases": results},
        "gradient": gradients,
        "effects": {
            "status": CoverageStatus.NOT_OBSERVABLE,
            "reason": "receipt has no separate alias/mutation observations; aggregate pass is not copied",
        },
        "execution": predicate.get("execution"),
        "policy": predicate.get("tolerance"),
    }

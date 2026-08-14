"""Contracts for stage-scoped source transitions and raw-case rebinding."""

from collections.abc import Mapping

import pytest
from pydantic import ValidationError

from sol_execbench.core.bench.performance_model.diagnostic_schema_versions import (
    DiagnosticArtifactSchema,
    DiagnosticSourceTransitionArtifactKind,
)
from sol_execbench.core.bench.performance_model.lifecycle.shared import (
    DiagnosticLifecycleArtifact,
)
from sol_execbench.core.bench.performance_model.models import WorkloadKind
from sol_execbench.core.bench.performance_model.source_transition import (
    DevelopmentCaseRebind,
    DiagnosticDevelopmentCaseRebindReceipt,
    DiagnosticSourceTransitionAttestation,
    QualificationTimeoutTransition,
    SemanticProjectionPair,
    SourcePathStageImpact,
    SourceStageDecision,
    SourceTransitionDisposition,
    SourceTransitionStage,
)


def _digest(value: int) -> str:
    return f"{value:064x}"


def _artifact(path: str, value: int) -> DiagnosticLifecycleArtifact:
    return DiagnosticLifecycleArtifact(
        relative_path=path,
        sha256=_digest(value),
        size_bytes=value,
    )


def _attestation_payload(
    *,
    projection_overrides: Mapping[str, SemanticProjectionPair] | None = None,
    changed_stage: SourceTransitionStage = SourceTransitionStage.QUALIFICATION_GPU,
) -> dict[str, object]:
    projections = {
        "policy_behavior_projection": SemanticProjectionPair(
            base_sha256=_digest(1), comparison_sha256=_digest(1)
        ),
        "design_behavior_projection": SemanticProjectionPair(
            base_sha256=_digest(2), comparison_sha256=_digest(2)
        ),
        "problems_tree_projection": SemanticProjectionPair(
            base_sha256=_digest(3), comparison_sha256=_digest(3)
        ),
        "raw_collection_projection": SemanticProjectionPair(
            base_sha256=_digest(4), comparison_sha256=_digest(4)
        ),
    }
    projections.update(projection_overrides or {})
    inventory = (_artifact("definition.json", 10),)
    reusable = tuple(
        _artifact(path, index)
        for index, path in enumerate(
            (
                "calibration/cal.audit.json",
                "calibration/cal.json",
                "corpus/design.json",
                "vram-policy.json",
            ),
            start=20,
        )
    )
    return {
        "base_source_revision": "1" * 40,
        "comparison_artifact_source_revision": "2" * 40,
        "target_source_revision": "3" * 40,
        "base_policy_path": "/evidence/r1/vram-policy.json",
        "comparison_policy_path": "/evidence/r2/vram-policy.json",
        "base_calibration_profile_path": "/evidence/r1/cal.json",
        "base_calibration_audit_path": "/evidence/r1/cal.audit.json",
        "base_design_path": "/evidence/r1/design.json",
        "comparison_design_path": "/evidence/r2/design.json",
        "base_problems_root": "/evidence/r1/problems",
        "comparison_problems_root": "/evidence/r2/problems",
        "git_patch_sha256": _digest(5),
        "source_changes": (
            SourcePathStageImpact(
                path="scripts/collector.py",
                change="modified",
                affected_stages=(changed_stage,),
                rationale="only the qualification watchdog changed",
            ),
        ),
        "stage_decisions": tuple(
            SourceStageDecision(
                stage=stage,
                disposition=(
                    SourceTransitionDisposition.CHANGED
                    if stage is changed_stage
                    else SourceTransitionDisposition.UNCHANGED
                ),
                rationale="reviewed exact diff",
            )
            for stage in SourceTransitionStage
        ),
        **projections,
        "qualification_timeout": QualificationTimeoutTransition(
            old_seconds=300, new_seconds=900
        ),
        "reusable_artifacts": reusable,
        "base_problems_inventory": inventory,
        "comparison_problems_inventory": inventory,
        "created_at": "2026-08-11T00:00:00+00:00",
    }


def _validate_attestation(
    payload: dict[str, object],
) -> DiagnosticSourceTransitionAttestation:
    return DiagnosticSourceTransitionAttestation.model_validate(
        {
            "schema_version": DiagnosticArtifactSchema.DIAGNOSTIC_SOURCE_TRANSITION,
            "artifact_kind": DiagnosticSourceTransitionArtifactKind.ATTESTATION,
            **payload,
        }
    )


def test_source_transition_accepts_timeout_only_semantic_reuse() -> None:
    attestation = _validate_attestation(_attestation_payload())

    assert attestation.raw_collection_projection.unchanged
    assert (
        next(
            item
            for item in attestation.stage_decisions
            if item.stage is SourceTransitionStage.QUALIFICATION_GPU
        ).disposition
        is SourceTransitionDisposition.CHANGED
    )


def test_source_transition_rejects_unaffected_decision_for_impacted_stage() -> (
    None
):
    payload = _attestation_payload()
    payload["stage_decisions"] = tuple(
        SourceStageDecision(
            stage=stage,
            disposition=SourceTransitionDisposition.UNCHANGED,
            rationale="incorrectly declared reusable",
        )
        for stage in SourceTransitionStage
    )

    with pytest.raises(ValidationError, match="disagrees with source diff"):
        _validate_attestation(payload)


@pytest.mark.parametrize(
    ("stage", "projection"),
    (
        (SourceTransitionStage.VRAM_POLICY, "policy_behavior_projection"),
        (SourceTransitionStage.DESIGN, "design_behavior_projection"),
        (SourceTransitionStage.RAW_COLLECTION, "raw_collection_projection"),
    ),
)
def test_unchanged_stage_requires_equal_semantic_projection(
    stage: SourceTransitionStage,
    projection: str,
) -> None:
    pair = SemanticProjectionPair(
        base_sha256=_digest(31), comparison_sha256=_digest(32)
    )
    payload = _attestation_payload(
        projection_overrides={projection: pair},
        changed_stage=SourceTransitionStage.QUALIFICATION_GPU,
    )

    with pytest.raises(ValidationError, match=f"unchanged {stage.value}"):
        _validate_attestation(payload)


def test_unchanged_design_requires_exact_problem_inventory() -> None:
    payload = _attestation_payload()
    payload["comparison_problems_inventory"] = (
        _artifact("definition.json", 11),
    )

    with pytest.raises(ValidationError, match="identical prepared problems"):
        _validate_attestation(payload)


def test_source_path_impact_requires_sorted_unique_stages() -> None:
    with pytest.raises(ValidationError, match="sorted and unique"):
        SourcePathStageImpact(
            path="collector.py",
            change="modified",
            affected_stages=(
                SourceTransitionStage.VRAM_POLICY,
                SourceTransitionStage.RAW_COLLECTION,
            ),
            rationale="out of order",
        )


def test_qualification_timeout_cannot_shrink() -> None:
    with pytest.raises(ValidationError, match="must increase"):
        QualificationTimeoutTransition(old_seconds=900, new_seconds=300)


def test_rebind_receipt_rejects_unsorted_case_ids() -> None:
    inventory = (_artifact("trace.jsonl", 40),)
    cases = tuple(
        DevelopmentCaseRebind(
            case_id=case_id,
            workload_kind=WorkloadKind.ELEMENTWISE,
            phase="point_fit",
            workload_uuid=f"diagnostic-{case_id}",
            evidence_manifest_sha256=_digest(index),
            inventory=inventory,
        )
        for index, case_id in enumerate(
            ("point_fit-elementwise-01", "point_fit-elementwise-00"), start=50
        )
    )

    with pytest.raises(ValidationError, match="sorted and unique"):
        DiagnosticDevelopmentCaseRebindReceipt(
            transition_attestation_sha256=_digest(60),
            base_source_revision="1" * 40,
            target_source_revision="2" * 40,
            source_design_sha256=_digest(61),
            target_design_sha256=_digest(62),
            source_root="/evidence/r2",
            target_root="/evidence/r1",
            qualification_root="/evidence/r1-qualification",
            qualification_gates=(
                _artifact("development/canary/gate.json", 63),
                _artifact("development/full/gate.json", 64),
                _artifact("static/gate.json", 65),
            ),
            cases=cases,
            created_at="2026-08-11T00:00:00+00:00",
        )

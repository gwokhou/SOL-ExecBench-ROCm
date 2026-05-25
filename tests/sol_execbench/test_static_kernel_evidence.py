from __future__ import annotations

import pytest
from pydantic import ValidationError

from sol_execbench.core.bench.static_kernel_evidence import (
    STATIC_KERNEL_EVIDENCE_SCHEMA_VERSION,
    StaticKernelEvidenceArtifact,
    StaticKernelEvidenceClassification,
    StaticKernelEvidenceReasonCode,
    StaticKernelEvidenceSidecar,
    StaticKernelEvidenceStatus,
    StaticKernelEvidenceToolRun,
    build_static_kernel_evidence_failed,
    build_static_kernel_evidence_partial,
    build_static_kernel_evidence_sidecar,
    build_static_kernel_evidence_skipped,
    build_static_kernel_evidence_unavailable,
    build_static_kernel_evidence_unsupported,
)


EXPECTED_STATUSES = {
    "collected",
    "partial",
    "unavailable",
    "unsupported",
    "failed",
    "skipped",
}

EXPECTED_REASON_CODES = {
    "static_evidence_not_requested",
    "static_evidence_collected",
    "partial_artifact_metadata",
    "partial_disassembly_only",
    "partial_metadata_only",
    "artifact_unavailable",
    "toolchain_unavailable",
    "unsupported_solution_type",
    "unsupported_architecture",
    "unsupported_artifact_type",
    "extractor_failed",
    "extractor_timeout",
    "parser_failed",
}


def _representative_sidecar() -> StaticKernelEvidenceSidecar:
    return build_static_kernel_evidence_sidecar(
        status=StaticKernelEvidenceStatus.COLLECTED,
        reason_code=StaticKernelEvidenceReasonCode.STATIC_EVIDENCE_COLLECTED,
        artifacts=[
            StaticKernelEvidenceArtifact(
                artifact_id="kernel-object",
                artifact_type="elf_object",
                status=StaticKernelEvidenceStatus.COLLECTED,
                reason_code=StaticKernelEvidenceReasonCode.STATIC_EVIDENCE_COLLECTED,
                source_path="build/kernel.o",
                size_bytes=4096,
                classification=StaticKernelEvidenceClassification(
                    metadata_present=True,
                    disassembly_present=True,
                    detected_architectures=["gfx1200", "gfx942"],
                    symbol_count=3,
                ),
            )
        ],
        tool_runs=[
            StaticKernelEvidenceToolRun(
                tool_id="llvm-objdump",
                command=["llvm-objdump", "--disassemble", "kernel.o"],
                status=StaticKernelEvidenceStatus.COLLECTED,
                reason_code=StaticKernelEvidenceReasonCode.STATIC_EVIDENCE_COLLECTED,
                returncode=0,
                stdout_tail="kernel symbols",
            )
        ],
    )


def test_static_kernel_evidence_round_trips_strict_json_payload():
    sidecar = _representative_sidecar()
    payload = sidecar.model_dump(mode="json")

    assert payload["schema_version"] == STATIC_KERNEL_EVIDENCE_SCHEMA_VERSION
    assert payload["schema_version"] == "sol_execbench.static_kernel_evidence.v1"
    assert StaticKernelEvidenceSidecar.model_validate(payload) == sidecar


def test_static_kernel_evidence_rejects_unknown_top_level_and_nested_fields():
    payload = _representative_sidecar().model_dump(mode="json")
    payload["unexpected"] = True

    with pytest.raises(ValidationError):
        StaticKernelEvidenceSidecar.model_validate(payload)

    nested_payload = _representative_sidecar().model_dump(mode="json")
    nested_payload["artifacts"][0]["unexpected"] = True

    with pytest.raises(ValidationError):
        StaticKernelEvidenceSidecar.model_validate(nested_payload)


def test_status_and_reason_code_vocabularies_are_locked():
    assert {status.value for status in StaticKernelEvidenceStatus} == EXPECTED_STATUSES
    assert {reason.value for reason in StaticKernelEvidenceReasonCode} == (
        EXPECTED_REASON_CODES
    )


def test_authority_fields_are_diagnostic_only_and_false_for_benchmark_truth():
    payload = _representative_sidecar().model_dump(mode="json")

    assert payload["diagnostic_only"] is True
    assert payload["correctness_authority"] is False
    assert payload["performance_authority"] is False
    assert payload["timing_authority"] is False
    assert payload["score_authority"] is False
    assert payload["paper_parity_authority"] is False
    assert payload["leaderboard_authority"] is False


@pytest.mark.parametrize(
    ("builder", "status", "reason_code"),
    [
        (
            build_static_kernel_evidence_skipped,
            "skipped",
            "static_evidence_not_requested",
        ),
        (
            build_static_kernel_evidence_unavailable,
            "unavailable",
            "toolchain_unavailable",
        ),
        (
            build_static_kernel_evidence_unsupported,
            "unsupported",
            "unsupported_solution_type",
        ),
        (build_static_kernel_evidence_failed, "failed", "extractor_failed"),
        (build_static_kernel_evidence_partial, "partial", "partial_artifact_metadata"),
    ],
)
def test_non_collected_helpers_return_full_sidecars_with_stable_empty_sections(
    builder,
    status,
    reason_code,
):
    sidecar = builder()
    payload = sidecar.model_dump(mode="json")

    assert StaticKernelEvidenceSidecar.model_validate(payload) == sidecar
    assert payload["status"] == status
    assert payload["reason_code"] == reason_code
    assert payload["artifacts"] == []
    assert payload["tool_runs"] == []
    assert payload["kernels"] == []
    assert payload["warnings"] == []
    assert payload["source_references"] == []


def test_artifact_entries_carry_per_artifact_status_reason_and_classification():
    payload = _representative_sidecar().model_dump(mode="json")
    artifact = payload["artifacts"][0]
    classification = artifact["classification"]

    assert artifact["status"] == "collected"
    assert artifact["reason_code"] == "static_evidence_collected"
    assert classification["metadata_present"] is True
    assert classification["disassembly_present"] is True
    assert classification["detected_architectures"] == ["gfx1200", "gfx942"]
    assert classification["symbol_count"] == 3

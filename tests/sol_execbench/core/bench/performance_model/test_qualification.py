"""Diagnostic corpus qualification contract tests."""

from sol_execbench.core.bench.batch_gpu_qualification import (
    BatchGPUQualificationStage,
)
from sol_execbench.core.bench.performance_model.diagnostic_schema_versions import (
    DiagnosticArtifactSchema,
    DiagnosticQualificationArtifactKind,
)
from sol_execbench.core.bench.performance_model.models import WorkloadKind
from sol_execbench.core.bench.performance_model.qualification import (
    DiagnosticCorpusQualification,
    DiagnosticQualificationReceipt,
)

_DIGEST = "a" * 64


def test_gate_and_receipt_share_one_discriminated_contract() -> None:
    receipt = DiagnosticQualificationReceipt(
        stage=BatchGPUQualificationStage.CANARY,
        role="development",
        family=WorkloadKind.MATMUL,
        case_ids=("case",),
        workload_uuids=("workload",),
        definition_sha256=_DIGEST,
        solution_sha256=_DIGEST,
        workload_sha256=_DIGEST,
        config_sha256=_DIGEST,
        trace_sha256=_DIGEST,
        log_sha256=_DIGEST,
        trace_count=1,
        created_at="2026-08-14T00:00:00Z",
    )
    gate = DiagnosticCorpusQualification(
        stage=BatchGPUQualificationStage.CANARY,
        role="development",
        design_sha256=_DIGEST,
        contract_sha256=_DIGEST,
        collector_sha256=_DIGEST,
        config_sha256=_DIGEST,
        preflight_sha256=_DIGEST,
        source_revision="revision",
        parent_gate_sha256=_DIGEST,
        case_ids=("case",),
        receipts=(receipt,),
        created_at="2026-08-14T00:00:00Z",
    )

    assert (
        gate.schema_version
        == DiagnosticArtifactSchema.DIAGNOSTIC_CORPUS_QUALIFICATION
    )
    assert gate.artifact_kind == DiagnosticQualificationArtifactKind.GATE
    assert receipt.schema_version == gate.schema_version
    assert receipt.artifact_kind == DiagnosticQualificationArtifactKind.RECEIPT

from __future__ import annotations

import json
from pathlib import Path

from sol_execbench.cli import static_evidence as cli_static_evidence
from sol_execbench.core.bench.static_kernel_evidence import (
    StaticKernelEvidenceArtifact,
    StaticKernelEvidenceReasonCode,
    StaticKernelEvidenceStatus,
    build_static_kernel_evidence_sidecar,
)


def test_static_evidence_paths_track_trace_output(tmp_path: Path):
    output = tmp_path / "run" / "trace.jsonl"
    staging = tmp_path / "staging"

    assert cli_static_evidence._static_evidence_directory(output, staging) == (
        tmp_path / "run" / "trace.jsonl.static-evidence"
    )
    assert cli_static_evidence._static_evidence_sidecar_path(output, staging) == (
        tmp_path / "run" / "trace.jsonl.static-evidence.json"
    )


def test_static_evidence_paths_fall_back_to_staging(tmp_path: Path):
    staging = tmp_path / "staging"

    assert cli_static_evidence._static_evidence_directory(None, staging) == (
        staging / "static-evidence"
    )
    assert cli_static_evidence._static_evidence_sidecar_path(None, staging) == (
        staging / "static-evidence.json"
    )


def test_static_evidence_sidecar_writes_summary(tmp_path: Path):
    output = tmp_path / "trace.jsonl"
    staging = tmp_path / "staging"
    sidecar = build_static_kernel_evidence_sidecar(
        status=StaticKernelEvidenceStatus.COLLECTED,
        reason_code=StaticKernelEvidenceReasonCode.STATIC_EVIDENCE_COLLECTED,
        artifacts=[
            StaticKernelEvidenceArtifact(
                artifact_id="artifact-benchmark_kernel.so",
                artifact_type="shared_library",
                status=StaticKernelEvidenceStatus.COLLECTED,
                reason_code=StaticKernelEvidenceReasonCode.STATIC_EVIDENCE_COLLECTED,
                persisted_path="artifacts/benchmark_kernel.so",
                inspectable=True,
            )
        ],
    )

    written = cli_static_evidence._write_static_evidence_sidecar(
        output, staging, sidecar
    )

    assert written == tmp_path / "trace.jsonl.static-evidence.json"
    assert written is not None
    payload = json.loads(written.read_text())
    assert payload["schema_version"] == "sol_execbench.static_kernel_evidence.v1"
    assert payload["summary"]["status"] == "collected"
    assert payload["summary"]["artifact_count"] == 1
    assert payload["summary"]["claim_boundaries"]["diagnostic_only"] is True
    assert payload["summary"]["claim_boundaries"]["score_authority"] is False


def test_static_evidence_none_does_not_collect(tmp_path: Path):
    sidecar = cli_static_evidence._collect_static_evidence_for_cli(
        enabled=cli_static_evidence.STATIC_EVIDENCE_NONE,
        is_cpp=True,
        staging_dir=tmp_path / "staging",
        output_file=tmp_path / "trace.jsonl",
    )

    assert sidecar is None


def test_static_evidence_auto_for_non_cpp_is_unsupported_sidecar(tmp_path: Path):
    sidecar = cli_static_evidence._collect_static_evidence_for_cli(
        enabled=cli_static_evidence.STATIC_EVIDENCE_AUTO,
        is_cpp=False,
        staging_dir=tmp_path / "staging",
        output_file=tmp_path / "trace.jsonl",
    )

    assert sidecar is not None
    assert sidecar.status == StaticKernelEvidenceStatus.UNSUPPORTED
    assert (
        sidecar.reason_code == StaticKernelEvidenceReasonCode.UNSUPPORTED_SOLUTION_TYPE
    )


def test_static_evidence_collection_failure_is_failed_sidecar(
    tmp_path: Path,
):
    def fail_collection(**kwargs):
        raise RuntimeError("collector failed")

    sidecar = cli_static_evidence._collect_static_evidence_for_cli(
        enabled=cli_static_evidence.STATIC_EVIDENCE_AUTO,
        is_cpp=True,
        staging_dir=tmp_path / "staging",
        output_file=tmp_path / "trace.jsonl",
        artifact_collector=fail_collection,
    )

    assert sidecar is not None
    assert sidecar.status == StaticKernelEvidenceStatus.FAILED
    assert sidecar.reason_code == StaticKernelEvidenceReasonCode.EXTRACTOR_FAILED
    assert sidecar.warnings[0].code == "static_evidence_collection_failed"
    assert "collector failed" in sidecar.warnings[0].message

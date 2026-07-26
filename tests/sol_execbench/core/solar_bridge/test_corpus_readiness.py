from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from sol_execbench.core.integrity import sha256_file
from sol_execbench.core.solar_bridge import corpus_readiness
from sol_execbench.core.solar_bridge.models import SolarStageAuditOutcome

ROOT = Path(__file__).resolve().parents[4]
MANIFEST = ROOT / "problems" / "AMD_AKA" / "manifest.yaml"


def _ready_outcome(request) -> SolarStageAuditOutcome:
    output = Path(request.output_dir)
    output.mkdir(parents=True)
    names = {
        "graph_extraction": "operator_graph.yaml",
        "einsum_conversion": "einsum_graph.yaml",
        "conversion_verification": "conversion-attestation.yaml",
    }
    stages = []
    for stage, name in names.items():
        path = output / name
        path.write_text(f"{stage}: passed\n", encoding="utf-8")
        stages.append(
            {
                "stage": stage,
                "status": "passed",
                "reason_code": None,
                "message": None,
                "artifact": {"path": name, "sha256": sha256_file(path)},
            }
        )
    return SolarStageAuditOutcome(
        status="ready",
        analysis_id=request.workload_uuid,
        output_dir=str(output),
        architecture_sha256=corpus_readiness.formal_architecture_profile_hash(),
        stages=tuple(stages),
    )


def test_corpus_audit_derives_and_addresses_the_full_scored_denominator(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        corpus_readiness,
        "run_solar_stage_worker",
        lambda request, **_kwargs: _ready_outcome(request),
    )

    result = corpus_readiness.audit_corpus_stage_readiness(
        MANIFEST,
        tmp_path / "audit",
    )

    assert result.ready
    assert result.problems == 35
    assert result.workloads == 122
    assert result.extraction_passed == 122
    assert result.conversion_passed == 122
    assert result.verification_passed == 122
    records = [
        json.loads(line)
        for line in result.matrix_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 122
    assert all(record["gfx_target"] == "gfx1200" for record in records)
    assert all(
        len(record["trace_identity_sha256"]) == 64
        and len(record["architecture_sha256"]) == 64
        for record in records
    )
    assert len({record["trace_identity_sha256"] for record in records}) == 122
    assert not any("l2n55" in record["problem_path"] for record in records)
    assert all(
        len(record["verification_seeds"]) == 3
        and set(record["verification_patterns"]) == {"random", "zeros", "boundary"}
        for record in records
    )
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["matrix"]["sha256"] == sha256_file(result.matrix_path)


def test_corpus_audit_keeps_failed_workload_in_the_matrix(
    tmp_path: Path,
    monkeypatch,
) -> None:
    failed_uuid = "aka-3267_doubled_matmul-w0"

    def audit(request, **_kwargs):
        if request.workload_uuid != failed_uuid:
            return _ready_outcome(request)
        return SolarStageAuditOutcome(
            status="failed",
            analysis_id=request.workload_uuid,
            failure_stage="einsum_conversion",
            reason_code="source_input_binding_failed",
            message="binding mismatch",
            stages=(
                {
                    "stage": "graph_extraction",
                    "status": "passed",
                    "artifact": None,
                },
                {
                    "stage": "einsum_conversion",
                    "status": "failed",
                    "reason_code": "source_input_binding_failed",
                },
                {"stage": "conversion_verification", "status": "not_run"},
            ),
        )

    monkeypatch.setattr(corpus_readiness, "run_solar_stage_worker", audit)
    result = corpus_readiness.audit_corpus_stage_readiness(
        MANIFEST,
        tmp_path / "audit",
    )

    assert not result.ready
    assert result.workloads == 122
    assert result.verification_passed == 121
    records = [
        json.loads(line)
        for line in result.matrix_path.read_text(encoding="utf-8").splitlines()
    ]
    failed = [record for record in records if record["workload_uuid"] == failed_uuid]
    assert len(failed) == 1
    assert failed[0]["reason_code"] == "source_input_binding_failed"
    assert len(failed[0]["trace_identity_sha256"]) == 64
    assert len(failed[0]["architecture_sha256"]) == 64


def test_corpus_audit_rejects_worker_architecture_identity_drift(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def wrong_architecture(request, **_kwargs):
        outcome = _ready_outcome(request)
        return replace(outcome, architecture_sha256="f" * 64)

    monkeypatch.setattr(
        corpus_readiness,
        "run_solar_stage_worker",
        wrong_architecture,
    )

    try:
        corpus_readiness.audit_corpus_stage_readiness(
            MANIFEST,
            tmp_path / "audit",
        )
    except ValueError as exc:
        assert str(exc) == "readiness worker architecture identity mismatch"
    else:
        raise AssertionError("architecture identity drift was accepted")


def test_corpus_audit_requires_every_stage_even_if_verification_passed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    failed_uuid = "aka-3267_doubled_matmul-w0"

    def inconsistent_stages(request, **_kwargs):
        outcome = _ready_outcome(request)
        if request.workload_uuid != failed_uuid:
            return outcome
        stages = list(outcome.stages)
        stages[0] = {
            **stages[0],
            "status": "failed",
            "reason_code": "graph_extraction_failed",
        }
        return replace(
            outcome,
            status="failed",
            stages=tuple(stages),
            failure_stage="graph_extraction",
            reason_code="graph_extraction_failed",
        )

    monkeypatch.setattr(
        corpus_readiness,
        "run_solar_stage_worker",
        inconsistent_stages,
    )

    result = corpus_readiness.audit_corpus_stage_readiness(
        MANIFEST,
        tmp_path / "audit",
    )

    assert not result.ready
    assert result.verification_passed == 122
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "incomplete"

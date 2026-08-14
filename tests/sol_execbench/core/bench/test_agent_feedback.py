from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from sol_execbench.core.bench.agent_feedback import (
    AgentFeedbackBuildIdentity,
    AgentFeedbackBuildRequest,
    AgentFeedbackSidecar,
    artifact_citation_from_path,
    build_agent_feedback_sidecar as _build_agent_feedback_sidecar,
)
from sol_execbench.core.bench.diagnostic_sidecar import (
    DiagnosticArtifactCitation,
    ExtendedDiagnosticIdentity,
)
from sol_execbench.core.bench.performance_model.schema_versions import (
    PerformanceArtifactSchema,
)
from sol_execbench.core.bench.rocm_profiler import (
    Rocprofv3ProfileResult,
    Rocprofv3ProfileStatus,
)
from sol_execbench.core.bench.static_kernel.evidence import (
    StaticKernelEvidenceSidecar,
)
from sol_execbench.core.data.trace import (
    Correctness,
    Environment,
    Evaluation,
    EvaluationStatus,
    Performance,
    Trace,
)
from sol_execbench.core.data.workload import ScalarInput, Workload


def build_agent_feedback_sidecar(
    *,
    traces: Sequence[Trace],
    profile_result: Rocprofv3ProfileResult | None = None,
    static_evidence: StaticKernelEvidenceSidecar | None = None,
    trace_path: str | None = None,
    target_id: str | None = None,
    run_id: str | None = None,
    candidate_id: str | None = None,
    source_sha256: str | None = None,
    sol_version: str | None = None,
    generated_at: str | None = None,
    artifact_citations: Sequence[DiagnosticArtifactCitation] = (),
) -> AgentFeedbackSidecar:
    """Keep individual builder tests focused on the field under assertion."""
    return _build_agent_feedback_sidecar(
        AgentFeedbackBuildRequest(
            traces=traces,
            profile_result=profile_result,
            static_evidence=static_evidence,
            identity=AgentFeedbackBuildIdentity(
                trace_path=trace_path,
                target_id=target_id,
                run_id=run_id,
                candidate_id=candidate_id,
                source_sha256=source_sha256,
                sol_version=sol_version,
                generated_at=generated_at,
            ),
            artifact_citations=artifact_citations,
        ),
    )


def _trace(status: EvaluationStatus = EvaluationStatus.PASSED) -> Trace:
    correctness = None
    performance = None
    if status in {
        EvaluationStatus.PASSED,
        EvaluationStatus.INCORRECT_NUMERICAL,
    }:
        correctness = Correctness(
            max_relative_error=0.0,
            max_absolute_error=0.0,
        )
    if status == EvaluationStatus.PASSED:
        performance = Performance(
            latency_ms=1.0,
            reference_latency_ms=2.0,
            speedup_factor=2.0,
        )
    return Trace(
        definition="toy",
        solution="candidate",
        workload=Workload(
            uuid="w0",
            axes={"n": 1},
            inputs={"n": ScalarInput(value=1)},
            checks=[{"type": "numeric", "output": "output"}],
        ),
        evaluation=Evaluation(
            status=status,
            environment=Environment(
                hardware="AMD gfx1200",
                libs={"hip": "7.0"},
            ),
            timestamp="2026-06-16T00:00:00Z",
            correctness=correctness,
            performance=performance,
        ),
    )


def test_agent_feedback_identity_uses_canonical_fields_only() -> None:
    identity = ExtendedDiagnosticIdentity(
        generated_at="2026-01-01T00:00:00Z",
        sol_version="v3.0.0",
        trace_path="trace.jsonl",
        target_id="gemm",
        run_id="run-1",
        candidate_id="candidate-sha",
        source_sha256="source-sha",
    )

    payload = identity.model_dump(mode="json", exclude_none=True)

    assert payload["sol_version"] == "v3.0.0"
    assert payload["candidate_id"] == "candidate-sha"
    assert payload["source_sha256"] == "source-sha"
    assert "sol_contract_version" not in payload
    assert "candidate_hash" not in payload
    assert "source_hash" not in payload


@pytest.mark.parametrize(
    ("legacy_alias", "legacy_kwargs"),
    [
        ("sol_contract_version", {"sol_contract_version": "v3.0.0"}),
        (
            "candidate_hash",
            {
                "sol_contract_version": "v3.0.0",
                "candidate_hash": "candidate-sha",
            },
        ),
        (
            "source_hash",
            {
                "sol_contract_version": "v3.0.0",
                "source_hash": "source-sha",
            },
        ),
    ],
)
def test_agent_feedback_identity_rejects_legacy_alias_fields(
    legacy_alias: str,
    legacy_kwargs: dict[str, str],
) -> None:
    with pytest.raises(ValidationError, match=legacy_alias):
        ExtendedDiagnosticIdentity(
            generated_at="2026-01-01T00:00:00Z",
            sol_version="v3.0.0",
            trace_path="trace.jsonl",
            candidate_id="candidate-sha",
            source_sha256="source-sha",
            **legacy_kwargs,
        )


def test_agent_feedback_sidecar_is_diagnostic_only_for_passing_trace():
    sidecar = build_agent_feedback_sidecar(traces=[_trace()])
    payload = sidecar.model_dump(mode="json")

    assert payload["schema_version"] == PerformanceArtifactSchema.AGENT_FEEDBACK
    assert payload["status"] == "available"
    assert payload["reason_code"] == "feedback_generated"
    assert payload["summary"]["status_counts"] == {"PASSED": 1}
    assert payload["authority"] == "diagnostic"
    assert payload["items"][0]["code"] == "all_evaluated_traces_passed"
    assert "Canonical Trace JSONL remains" in payload["limitations"][1]


def test_agent_feedback_sidecar_builder_emits_no_legacy_identity_aliases() -> (
    None
):
    sidecar = build_agent_feedback_sidecar(
        traces=[_trace()],
        trace_path="trace.jsonl",
        target_id="gemm",
        run_id="run-1",
        candidate_id="candidate-sha",
        source_sha256="source-sha",
        sol_version="v3.0.0",
    )

    raw_identity = sidecar.to_dict()["identity"]
    assert isinstance(raw_identity, dict)
    identity = cast(dict[str, object], raw_identity)

    assert identity["candidate_id"] == "candidate-sha"
    assert identity["source_sha256"] == "source-sha"
    assert identity["sol_version"] == "v3.0.0"
    assert "candidate_hash" not in identity
    assert "source_hash" not in identity
    assert "sol_contract_version" not in identity


def test_agent_feedback_sidecar_records_identity_and_artifact_citations(
    tmp_path: Path,
):
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text('{"status":"ok"}\n')
    citation = artifact_citation_from_path(
        kind="trace",
        label="canonical_trace_jsonl",
        path=trace_path,
    )

    sidecar = build_agent_feedback_sidecar(
        traces=[_trace()],
        trace_path=str(trace_path),
        target_id="problem-0",
        run_id="run-0",
        candidate_id="candidate-sha",
        source_sha256="source-sha",
        sol_version="v3.0.0",
        generated_at="2026-06-16T00:00:00Z",
        artifact_citations=[citation],
    )
    payload = sidecar.model_dump(mode="json")

    assert payload["identity"]["generated_at"] == "2026-06-16T00:00:00Z"
    assert payload["identity"]["trace_path"] == "trace.jsonl"
    assert payload["identity"]["target_id"] == "problem-0"
    assert payload["identity"]["run_id"] == "run-0"
    assert payload["identity"]["sol_version"] == "v3.0.0"
    assert payload["identity"]["candidate_id"] == "candidate-sha"
    assert payload["identity"]["source_sha256"] == "source-sha"
    assert "sol_contract_version" not in payload["identity"]
    assert "candidate_hash" not in payload["identity"]
    assert "source_hash" not in payload["identity"]
    assert payload["artifact_citations"] == [
        {
            "kind": "trace",
            "label": "canonical_trace_jsonl",
            "path": "trace.jsonl",
            "sha256": citation.sha256,
            "status": None,
        },
    ]
    assert citation.sha256 is not None
    assert len(citation.sha256) == 64


def test_agent_feedback_sidecar_summarizes_failures_and_optional_profile():
    profile = Rocprofv3ProfileResult(
        status=Rocprofv3ProfileStatus.UNAVAILABLE,
        command=("rocprofv3", "--", "python", "eval_driver.py"),
        output_directory=Path("profile"),
        output_file="profile",
        skipped_reason="rocprofv3 missing",
        profiler_available=False,
    )

    sidecar = build_agent_feedback_sidecar(
        traces=[_trace(EvaluationStatus.COMPILE_ERROR)],
        profile_result=profile,
    )
    payload = sidecar.model_dump(mode="json")

    assert payload["status"] == "partial"
    assert payload["reason_code"] == "partial_diagnostics"
    assert payload["summary"]["profile_status"] == "unavailable"
    assert any(ref["kind"] == "profile" for ref in payload["source_refs"])


def test_agent_feedback_sidecar_includes_trace_feedback_items() -> None:
    sidecar = build_agent_feedback_sidecar(
        traces=[_trace(EvaluationStatus.COMPILE_ERROR)],
    )
    payload = sidecar.model_dump(mode="json")

    assert payload["items"][0]["code"] == "compile_error"
    assert payload["items"][0]["bottleneck"] == "compile_failure"


def test_agent_feedback_sidecar_rejects_authority_override():
    sidecar = build_agent_feedback_sidecar(traces=[_trace()])
    payload = sidecar.model_dump(mode="json")
    payload["authority"] = "score"

    with pytest.raises(ValidationError):
        type(sidecar).model_validate(payload)


def test_agent_feedback_authority_freezes_diagnostic_boundary():
    sidecar = build_agent_feedback_sidecar(traces=[_trace()])
    payload = sidecar.model_dump(mode="json")

    assert payload["authority"] == "diagnostic"
    payload["authority"] = "none"

    with pytest.raises(ValidationError):
        type(sidecar).model_validate(payload)


def test_agent_feedback_sidecar_rejects_unknown_bottleneck():
    sidecar = build_agent_feedback_sidecar(
        traces=[_trace(EvaluationStatus.COMPILE_ERROR)],
    )
    payload = sidecar.model_dump(mode="json")
    payload["items"][0]["bottleneck"] = "ad_hoc_bottleneck"

    with pytest.raises(ValidationError):
        type(sidecar).model_validate(payload)

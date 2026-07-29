from __future__ import annotations

import json
from pathlib import Path

import yaml
from click.testing import CliRunner

from sol_execbench.cli.main import cli
from sol_execbench.core.bench.agent_feedback import (
    AgentFeedbackBuildRequest,
    build_agent_feedback_sidecar,
)
from sol_execbench.core.bench.diagnostic_sidecar import (
    DiagnosticIdentity,
    DiagnosticSidecarStatus,
    SizedDiagnosticArtifactCitation,
)
from sol_execbench.core.bench.performance_model.governance import (
    evaluate_performance_diagnostic_governance,
    validate_performance_diagnostic_freshness,
)
from sol_execbench.core.bench.performance_model.models import (
    PerformanceDiagnosticSidecar,
)
from sol_execbench.core.bench.profile_summary import (
    ProfileSummaryContent,
    ProfileSummaryReasonCode,
    ProfileSummarySidecar,
)
from sol_execbench.core.bench.static_kernel.evidence import (
    StaticISAAnalysis,
    StaticKernelEvidenceArtifact,
    StaticKernelEvidenceKernel,
    StaticKernelEvidenceReasonCode,
    StaticKernelEvidenceSidecar,
    StaticKernelEvidenceStatus,
)
from sol_execbench.core.data.json_utils import (
    atomic_write_json_value,
    atomic_write_jsonl_values,
    load_json_file,
)
from sol_execbench.core.data.trace import (
    Correctness,
    Environment,
    Evaluation,
    EvaluationStatus,
    Performance,
    Trace,
)
from sol_execbench.core.data.workload import Workload
from sol_execbench.core.integrity import sha256_file
from sol_execbench.core.solar_bridge.performance import (
    SOLAR_ANALYSIS_SCHEMA_VERSION,
)


def _trace(tmp_path: Path) -> tuple[Path, Trace]:
    sample = (
        Path(__file__).parents[2]
        / "samples"
        / "rdna4_vecadd"
        / "workload.jsonl"
    )
    workload = Workload.model_validate_json(
        sample.read_text(encoding="utf-8").splitlines()[0],
    )
    trace = Trace(
        definition="vector_add",
        workload=workload,
        solution="candidate",
        evaluation=Evaluation(
            status=EvaluationStatus.PASSED,
            environment=Environment(
                hardware="gfx1200",
                libs={"rocm": "7.2"},
                clocks_locked=True,
            ),
            timestamp="2026-07-29T00:00:00Z",
            correctness=Correctness(),
            performance=Performance(latency_ms=0.25),
        ),
    )
    path = tmp_path / "trace.jsonl"
    atomic_write_jsonl_values(path, [trace])
    return path, trace


def _solar(path: Path) -> None:
    payload = {
        "schema_version": SOLAR_ANALYSIS_SCHEMA_VERSION,
        "layers": {
            "add": {
                "type": "add",
                "tensor_shapes": {
                    "inputs": [[64], [64]],
                    "outputs": [[64]],
                },
            },
        },
        "total": {
            "flops": 64,
            "resource_work": {"valu": {"fp32": 64}},
            "prefetched_bytes": 768,
            "lower_bound_seconds": 1.0e-6,
        },
        "metadata": {
            "fusion": {"regions": [{"id": "region-0", "layers": ["add"]}]},
        },
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def _profile(path: Path, trace_path: Path) -> None:
    sidecar = ProfileSummarySidecar(
        status=DiagnosticSidecarStatus.PARTIAL,
        reason_code=ProfileSummaryReasonCode.PROFILE_PARTIAL,
        identity=DiagnosticIdentity(
            generated_at="2026-07-29T00:00:00Z",
            sol_version="4.0.0",
            trace_path=trace_path.name,
            run_id=sha256_file(trace_path),
        ),
        summary=ProfileSummaryContent(artifact_count=0),
        artifact_citations=[
            SizedDiagnosticArtifactCitation(
                kind="trace",
                label="canonical_trace_jsonl",
                path=trace_path.name,
                sha256=sha256_file(trace_path),
                size_bytes=trace_path.stat().st_size,
            ),
        ],
    )
    atomic_write_json_value(path, sidecar.to_dict())


def _static(path: Path) -> None:
    code_hash = "c" * 64
    sidecar = StaticKernelEvidenceSidecar(
        status=StaticKernelEvidenceStatus.COLLECTED,
        reason_code=StaticKernelEvidenceReasonCode.STATIC_EVIDENCE_COLLECTED,
        artifacts=[
            StaticKernelEvidenceArtifact(
                artifact_id="code-object",
                artifact_type="rocm_binary",
                status=StaticKernelEvidenceStatus.COLLECTED,
                sha256=code_hash,
                target_architecture="gfx1200",
                inspectable=True,
            ),
        ],
        kernels=[StaticKernelEvidenceKernel(name="vector_add")],
        isa_analyses=[
            StaticISAAnalysis(
                artifact_id="code-object",
                architecture="gfx1200",
                status=StaticKernelEvidenceStatus.COLLECTED,
                decoded_instruction_count=2,
                functional_group_counts={"VALU": 2},
                code_object_sha256=code_hash,
            ),
        ],
    )
    atomic_write_json_value(path, sidecar.to_dict())


def test_performance_diagnostics_cli_and_agent_feedback_governance(
    tmp_path: Path,
) -> None:
    trace_path, trace = _trace(tmp_path)
    solar_path = tmp_path / "solar-analysis.yaml"
    profile_path = tmp_path / "trace.profile-summary.json"
    static_path = tmp_path / "trace.static-evidence.json"
    output = tmp_path / "trace.performance-diagnostic.json"
    _solar(solar_path)
    _profile(profile_path, trace_path)
    _static(static_path)

    result = CliRunner().invoke(
        cli,
        [
            "--format",
            "json",
            "diagnostics",
            "performance",
            "--trace",
            str(trace_path),
            "--solar-analysis",
            f"{trace.workload.uuid}={solar_path}",
            "--profile-summary",
            str(profile_path),
            "--static-evidence",
            str(static_path),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    response = json.loads(result.output)
    assert response["ok"] is True
    assert response["data"]["diagnostic_only"] is True
    assert response["artifacts"] == [
        {
            "path": str(output),
            "type": "performance_diagnostic_json",
        },
    ]
    diagnostic = load_json_file(PerformanceDiagnosticSidecar, output)
    assert diagnostic.status is DiagnosticSidecarStatus.PARTIAL
    assert diagnostic.workloads[0].t_measured_ms == 0.25
    assert diagnostic.workloads[0].ratios[0].reason_codes == [
        "trusted_frontier_unavailable"
    ]

    freshness = validate_performance_diagnostic_freshness(
        diagnostic,
        run_id=diagnostic.run_id,
        candidate_sha256=diagnostic.candidate_sha256,
        gpu_architecture="gfx1200",
        trace_sha256=sha256_file(trace_path),
    )
    governance = evaluate_performance_diagnostic_governance(
        sidecar=diagnostic,
        freshness=freshness,
    )
    feedback = build_agent_feedback_sidecar(
        AgentFeedbackBuildRequest(
            traces=[trace],
            performance_diagnostic=diagnostic,
            performance_governance=governance,
        ),
    )
    assert "model_gap_no_kernel_action" in {
        item.code for item in feedback.items
    }

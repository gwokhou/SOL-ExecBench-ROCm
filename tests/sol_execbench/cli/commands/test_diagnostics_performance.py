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
from sol_execbench.core.bench.performance_model.access_evidence import (
    PerformanceAccessEvidenceSidecar,
    WorkloadAccessEvidence,
)
from sol_execbench.core.bench.performance_model.evidence_manifest import (
    PerformanceEvidenceArtifact,
    PerformanceEvidenceArtifactKind,
    PerformanceEvidenceManifest,
    PerformanceRunIdentity,
)
from sol_execbench.core.bench.performance_model.governance import (
    evaluate_performance_diagnostic_governance,
    validate_performance_diagnostic_freshness,
)
from sol_execbench.core.bench.performance_model.models import (
    PerformanceDiagnosticSidecar,
)
from sol_execbench.core.bench.performance_model.timing_evidence import (
    PerformanceTimingEvidenceSidecar,
    WorkloadTimingEvidence,
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
from sol_execbench.core.integrity import sha256_file, stable_json_checksum
from sol_execbench.core.solar_bridge.performance import (
    SOLAR_ANALYSIS_SCHEMA_VERSION,
    SOLAR_REQUEST_MANIFEST_SCHEMA_VERSION,
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
                "tensor_dtypes": {
                    "inputs": ["float32", "float32"],
                    "outputs": ["float32"],
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


def _solar_manifest(path: Path, analysis: Path, trace: Trace) -> None:
    payload = {
        "schema_version": SOLAR_REQUEST_MANIFEST_SCHEMA_VERSION,
        "analysis_id": f"{trace.definition}:{trace.workload.uuid}",
        "architecture_sha256": "a" * 64,
        "reference": {"name": "test", "sha256": "b" * 64},
        "analysis_contract": {
            "ir_path": "torchview_extended_einsum",
            "extraction_kind": "torchview",
            "precision": "fp32",
            "ir_kind": "extended_einsum",
            "trace_seed": 200,
            "verification_seeds": [11, 29, 47],
            "atol": 1e-2,
            "rtol": 1e-2,
            "required_matched_ratio": 0.99,
            "max_error_cap": None,
            "allow_negative_inf": False,
            "preserved_input_indices": [],
            "require_orojenesis": True,
        },
        "sol_score_eligible": True,
        "publication_eligible": True,
        "artifacts": [
            {
                "path": "solar-analysis.yaml",
                "sha256": sha256_file(analysis),
            }
        ],
        "bound": {
            "seconds": 1e-6,
            "kind": "capacity_constrained_tile_aware_v1",
            "limiting_resource": "valu",
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


def _timing(path: Path, trace_path: Path, trace: Trace) -> str:
    solution_hash = "e" * 64
    sidecar = PerformanceTimingEvidenceSidecar(
        run_id=sha256_file(trace_path),
        trace_sha256=sha256_file(trace_path),
        solution_sha256=solution_hash,
        workloads=[
            WorkloadTimingEvidence(
                workload_uuid=trace.workload.uuid,
                input_sha256="a" * 64,
                latency_ms=0.25,
                lower_ms=0.24,
                upper_ms=0.26,
                trial_samples_ms=[[0.24, 0.25, 0.26]],
                warmup_runs=3,
                timing_protocol="device_event_v1",
            )
        ],
    )
    atomic_write_json_value(path, sidecar.to_dict())
    return solution_hash


def _access(path: Path, trace_path: Path, trace: Trace) -> None:
    sidecar = PerformanceAccessEvidenceSidecar(
        status=DiagnosticSidecarStatus.AVAILABLE,
        run_id=sha256_file(trace_path),
        trace_sha256=sha256_file(trace_path),
        workloads=[
            WorkloadAccessEvidence(
                workload_uuid=trace.workload.uuid,
                canonical_input_sha256="a" * 64,
                patterns=[],
            )
        ],
    )
    atomic_write_json_value(path, sidecar.to_dict())


def _evidence_manifest(
    path: Path,
    *,
    trace_path: Path,
    trace: Trace,
    timing_path: Path,
    access_path: Path,
    profile_path: Path,
    static_path: Path,
    solution_hash: str,
) -> None:
    provenance = path.parent / "counter-metadata.json"
    counter = path.parent / "pass_1.csv"
    rocpd = path.parent / "profile.rocpd"
    atomic_write_json_value(
        provenance,
        {
            "schema_version": "sol_execbench.rocprofv3_counter_provenance.v5",
            "diagnostic_only": True,
            "score_authority": False,
            "replay_phase": "evidence",
        },
    )
    counter.write_text(
        "Dispatch_Id,Queue_Id,Grid_Size,Kernel_Name,Workgroup_Size,"
        "Counter_Name,Counter_Value\n"
        "1,0,64,vector_add,64,SQ_WAVES,1\n",
        encoding="utf-8",
    )
    rocpd.write_bytes(b"rocpd")
    artifact_paths = [
        (PerformanceEvidenceArtifactKind.TRACE, trace_path),
        (PerformanceEvidenceArtifactKind.TIMING, timing_path),
        (PerformanceEvidenceArtifactKind.ACCESS_PATTERN, access_path),
        (PerformanceEvidenceArtifactKind.PROFILE_SUMMARY, profile_path),
        (PerformanceEvidenceArtifactKind.STATIC_EVIDENCE, static_path),
        (PerformanceEvidenceArtifactKind.COUNTER_PROVENANCE, provenance),
        (PerformanceEvidenceArtifactKind.COUNTER_CSV, counter),
        (PerformanceEvidenceArtifactKind.ROCPD, rocpd),
    ]
    manifest = PerformanceEvidenceManifest(
        status=DiagnosticSidecarStatus.PARTIAL,
        identity=PerformanceRunIdentity(
            run_id=sha256_file(trace_path),
            definition=trace.definition,
            definition_sha256=stable_json_checksum(trace.definition),
            workload_uuid=trace.workload.uuid,
            workload_sha256=stable_json_checksum(
                trace.workload.model_dump(mode="json")
            ),
            solution_sha256=solution_hash,
            candidate_sha256="f" * 64,
            gpu_architecture="gfx1200",
            clock_mode="locked",
            timing_protocol="device_event_v1",
        ),
        artifacts=[
            PerformanceEvidenceArtifact(
                kind=kind,
                path=artifact_path.name,
                sha256=sha256_file(artifact_path),
                size_bytes=artifact_path.stat().st_size,
            )
            for kind, artifact_path in artifact_paths
        ],
        code_object_sha256=["c" * 64],
        reason_codes=["gpu_identity_missing"],
    )
    atomic_write_json_value(path, manifest.to_dict())


def test_performance_diagnostics_cli_and_agent_feedback_governance(
    tmp_path: Path,
) -> None:
    trace_path, trace = _trace(tmp_path)
    solar_path = tmp_path / "solar-analysis.yaml"
    solar_manifest_path = tmp_path / "solar-manifest.yaml"
    profile_path = tmp_path / "trace.profile-summary.json"
    static_path = tmp_path / "trace.static-evidence.json"
    timing_path = tmp_path / "trace.performance-timing.json"
    access_path = tmp_path / "trace.performance-access.json"
    evidence_path = tmp_path / "trace.performance-evidence.json"
    output = tmp_path / "trace.performance-diagnostic.json"
    _solar(solar_path)
    _solar_manifest(solar_manifest_path, solar_path, trace)
    _profile(profile_path, trace_path)
    _static(static_path)
    solution_hash = _timing(timing_path, trace_path, trace)
    _access(access_path, trace_path, trace)
    _evidence_manifest(
        evidence_path,
        trace_path=trace_path,
        trace=trace,
        timing_path=timing_path,
        access_path=access_path,
        profile_path=profile_path,
        static_path=static_path,
        solution_hash=solution_hash,
    )

    result = CliRunner().invoke(
        cli,
        [
            "--format",
            "json",
            "diagnostics",
            "performance",
            "--evidence-manifest",
            str(evidence_path),
            "--solar-manifest",
            str(solar_manifest_path),
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

    feedback_output = tmp_path / "trace.performance-agent-feedback.json"
    feedback_result = CliRunner().invoke(
        cli,
        [
            "--format",
            "json",
            "diagnostics",
            "agent-feedback",
            "--performance-diagnostic",
            str(output),
            "--evidence-manifest",
            str(evidence_path),
            "--output",
            str(feedback_output),
        ],
    )

    assert feedback_result.exit_code == 0, feedback_result.output
    feedback_response = json.loads(feedback_result.output)
    assert feedback_response["ok"] is True
    assert feedback_response["data"]["diagnostic_only"] is True
    assert feedback_output.is_file()

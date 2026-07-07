from __future__ import annotations

import json
import subprocess
from pathlib import Path


from sol_execbench.cli import evaluation as cli_evaluation
from sol_execbench.cli import sidecars as cli_sidecars
from sol_execbench.core.bench.rocm_profiler import Rocprofv3ProfileResult
from sol_execbench.core.bench.rocm_profiler import Rocprofv3ProfileArtifact
from sol_execbench.core.bench.static_kernel_evidence import (
    StaticKernelEvidenceArtifact,
    StaticKernelEvidenceReasonCode,
    StaticKernelEvidenceStatus,
    build_static_kernel_evidence_sidecar,
)
from sol_execbench.core.data.solution import (
    BuildSpec,
    Solution,
    SourceFile,
    SupportedHardware,
    SupportedLanguages,
)
from sol_execbench.core.data.trace import (
    Environment,
    Evaluation,
    EvaluationStatus,
    Trace,
)
from sol_execbench.core.data.workload import ScalarInput
from sol_execbench.core.data.workload import Workload


def _solution(source: str = "def run(x):\n    return x\n") -> Solution:
    return Solution(
        name="candidate",
        definition="toy",
        author="agent",
        spec=BuildSpec(
            languages=[SupportedLanguages.PYTORCH],
            target_hardware=[SupportedHardware.LOCAL],
            entry_point="solution.py::run",
        ),
        sources=[SourceFile(path="solution.py", content=source)],
    )


def test_run_evaluation_command_passes_flashinfer_env(tmp_path: Path, monkeypatch):
    captured_env = None

    def fake_env(base_env):
        env = dict(base_env)
        env["FLASHINFER_TRACE_DIR"] = "/repo"
        return env

    def fake_run(*args, **kwargs):
        nonlocal captured_env
        captured_env = kwargs["env"]
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout="",
            stderr="",
        )

    result = cli_evaluation._run_evaluation_command(
        ["python", "eval_driver.py"],
        staging_dir=tmp_path,
        timeout=30,
        env_builder=fake_env,
        runner=fake_run,
    )

    assert result.returncode == 0
    assert captured_env is not None
    assert captured_env["PYTORCH_ALLOC_CONF"] == "expandable_segments:True"
    assert captured_env["FLASHINFER_TRACE_DIR"] == "/repo"


def test_run_profiled_evaluation_requests_graceful_eval_driver_exit(
    tmp_path: Path, monkeypatch
):
    captured_env = None

    def fake_env(base_env):
        env = dict(base_env)
        env["FLASHINFER_TRACE_DIR"] = "/repo"
        return env

    def fake_run(command, **kwargs):
        nonlocal captured_env
        captured_env = kwargs["env"]
        output_dir = Path(command[command.index("--output-directory") + 1])
        output_file = command[command.index("--output-file") + 1]
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"{output_file}_results.db").write_text("profile db")
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout='{"definition": "demo"}\n',
            stderr="",
        )

    profiled_proc, profile_result = cli_evaluation._run_profiled_evaluation(
        ["python", "eval_driver.py"],
        staging_dir=tmp_path,
        output_file=tmp_path / "trace.jsonl",
        timeout=30,
        env_builder=fake_env,
        subprocess_run=fake_run,
        rocprofv3_available=True,
    )

    assert profiled_proc is not None
    assert profile_result.succeeded is True
    assert captured_env is not None
    assert captured_env["PYTORCH_ALLOC_CONF"] == "expandable_segments:True"
    assert captured_env["FLASHINFER_TRACE_DIR"] == "/repo"
    assert captured_env["SOL_EXECBENCH_GRACEFUL_EXIT"] == "1"


def test_no_trace_diagnostics_sidecar_uses_trace_output_path(tmp_path: Path):
    output = tmp_path / "traces.jsonl"
    staging = tmp_path / "staging"

    sidecar = cli_evaluation._no_trace_diagnostics_sidecar_path(
        output,
        staging,
        keep_staging=False,
    )

    assert sidecar == tmp_path / "traces.jsonl.no-trace-diagnostics.json"


def test_no_trace_diagnostics_sidecar_survives_removed_staging(tmp_path: Path):
    staging = tmp_path / "sol_execbench_demo"

    sidecar = cli_evaluation._no_trace_diagnostics_sidecar_path(
        None,
        staging,
        keep_staging=False,
    )

    assert sidecar.parent != staging
    assert sidecar.name == "sol_execbench_demo.no-trace-diagnostics.json"


def test_no_trace_diagnostics_sidecar_keeps_staging_when_requested(tmp_path: Path):
    staging = tmp_path / "sol_execbench_demo"

    sidecar = cli_evaluation._no_trace_diagnostics_sidecar_path(
        None,
        staging,
        keep_staging=True,
    )

    assert sidecar == staging / "no-trace-diagnostics.json"


def test_no_trace_diagnostics_sidecar_records_bounded_failure_output(tmp_path: Path):
    output = tmp_path / "traces.jsonl"
    staging = tmp_path / "staging"
    stdout = "library noise\n" + ("x" * (cli_evaluation._DIAGNOSTIC_TAIL_LIMIT + 10))
    stderr = "runtime failed\n" + ("y" * (cli_evaluation._DIAGNOSTIC_TAIL_LIMIT + 20))

    written = cli_evaluation._write_no_trace_diagnostics_sidecar(
        output_file=output,
        staging_dir=staging,
        keep_staging=False,
        reason="no_parseable_traces",
        returncode=2,
        stdout=stdout,
        stderr=stderr,
    )

    assert written == tmp_path / "traces.jsonl.no-trace-diagnostics.json"
    assert written is not None
    payload = json.loads(written.read_text())
    assert (
        payload["schema_version"] == cli_evaluation.NO_TRACE_DIAGNOSTICS_SCHEMA_VERSION
    )
    assert payload["diagnostic_only"] is True
    assert payload["canonical_trace_jsonl"] is False
    assert payload["reason"] == "no_parseable_traces"
    assert payload["returncode"] == 2
    assert payload["stdout_tail"] == stdout[-cli_evaluation._DIAGNOSTIC_TAIL_LIMIT :]
    assert payload["stderr_tail"] == stderr[-cli_evaluation._DIAGNOSTIC_TAIL_LIMIT :]
    assert payload["stdout_truncated"] is True
    assert payload["stderr_truncated"] is True
    assert payload["stdout_line_count"] == 2
    assert payload["stderr_line_count"] == 2


def test_no_trace_diagnostics_sidecar_records_empty_stdout_failure(tmp_path: Path):
    output = tmp_path / "traces.jsonl"
    staging = tmp_path / "staging"

    written = cli_evaluation._write_no_trace_diagnostics_sidecar(
        output_file=output,
        staging_dir=staging,
        keep_staging=False,
        reason="evaluation_failed_no_stdout",
        returncode=1,
        stdout="",
        stderr="Traceback: boom",
    )

    assert written is not None
    payload = json.loads(written.read_text())
    assert payload["reason"] == "evaluation_failed_no_stdout"
    assert payload["stdout_tail"] == ""
    assert payload["stderr_tail"] == "Traceback: boom"
    assert payload["stdout_truncated"] is False
    assert payload["stderr_truncated"] is False


def test_no_trace_diagnostics_filters_benign_amdgpu_ids_noise(tmp_path: Path):
    output = tmp_path / "traces.jsonl"
    staging = tmp_path / "staging"

    written = cli_evaluation._write_no_trace_diagnostics_sidecar(
        output_file=output,
        staging_dir=staging,
        keep_staging=False,
        reason="no_parseable_traces",
        returncode=1,
        stdout="",
        stderr=(
            "/opt/amdgpu/share/libdrm/amdgpu.ids: No such file or directory\n"
            "real stderr\n"
        ),
    )

    assert written is not None
    payload = json.loads(written.read_text())
    assert "amdgpu.ids" not in payload["stderr_tail"]
    assert payload["stderr_tail"] == "real stderr\n"
    assert payload["stderr_line_count"] == 1


def test_profile_sidecar_is_disabled_when_no_profile_result(tmp_path: Path):
    output = tmp_path / "trace.jsonl"

    written = cli_sidecars._write_profile_sidecar(output, None)

    assert written is None
    assert not (tmp_path / "trace.jsonl.profile.json").exists()


def test_profile_sidecar_records_diagnostic_metadata(tmp_path: Path):
    output = tmp_path / "trace.jsonl"
    result = Rocprofv3ProfileResult(
        status="unavailable",
        command=("rocprofv3", "--", "python", "eval_driver.py"),
        output_directory=tmp_path / "trace.jsonl.rocprofv3",
        output_file="profile",
        skipped_reason="rocprofv3 is not available on PATH",
        profiler_available=False,
    )

    written = cli_sidecars._write_profile_sidecar(output, result)

    assert written == tmp_path / "trace.jsonl.profile.json"
    assert written is not None
    payload = json.loads(written.read_text())
    assert payload["schema_version"] == "sol_execbench.rocprofv3_profile.v1"
    assert payload["status"] == "unavailable"
    assert payload["diagnostic_only"] is True
    assert payload["score_authority"] is False
    assert payload["skipped_reason"] == "rocprofv3 is not available on PATH"


def test_profile_summary_sidecar_tracks_trace_output(tmp_path: Path):
    output = tmp_path / "trace.jsonl"

    assert cli_sidecars._profile_summary_sidecar_path(output) == (
        tmp_path / "trace.jsonl.profile-summary.json"
    )
    assert cli_sidecars._profile_summary_sidecar_path(None) is None


def test_profile_summary_sidecar_records_bounded_metadata(tmp_path: Path):
    output = tmp_path / "trace.jsonl"
    output.write_text('{"definition":"toy"}\n')
    profile_metadata = tmp_path / "trace.jsonl.profile.json"
    profile_metadata.write_text(
        '{"schema_version":"sol_execbench.rocprofv3_profile.v1"}\n'
    )
    profile_artifact_dir = tmp_path / "trace.jsonl.rocprofv3" / "trace"
    profile_artifact_dir.mkdir(parents=True)
    profile_artifact = profile_artifact_dir / "trace.rocpd"
    profile_artifact.write_text("profile artifact\n")
    counter_artifact = profile_artifact_dir / "trace_counters.csv"
    counter_artifact.write_text("Metric,Value,Unit\nSQ_INSTS_VALU,12,count\n")
    result = Rocprofv3ProfileResult(
        status="success",
        command=("rocprofv3", "--kernel-trace", "--", "python", "eval_driver.py"),
        output_directory=tmp_path / "trace.jsonl.rocprofv3",
        output_file="trace",
        artifacts=(
            Rocprofv3ProfileArtifact(
                path=profile_artifact,
                kind="rocpd",
                size_bytes=profile_artifact.stat().st_size,
            ),
            Rocprofv3ProfileArtifact(
                path=counter_artifact,
                kind="counter_csv",
                size_bytes=counter_artifact.stat().st_size,
            ),
        ),
        returncode=0,
        profiler_available=True,
        artifact_coverage_status="complete",
        reason_codes=("rocprof_artifacts_registered",),
    )

    written = cli_sidecars._write_profile_summary_sidecar(
        output,
        result,
        profile_sidecar_path=profile_metadata,
        run_id="run-001",
    )

    assert written == tmp_path / "trace.jsonl.profile-summary.json"
    assert written is not None
    payload = json.loads(written.read_text())
    assert payload["schema_version"] == "sol_execbench.profile_summary.v2"
    assert payload["status"] == "available"
    assert payload["authority"] == "diagnostic"
    assert payload["identity"]["trace_path"] == "trace.jsonl"
    assert payload["identity"]["run_id"] == "run-001"
    assert payload["identity"]["sol_version"] == "v1.42"
    assert "sol_contract_version" not in payload["identity"]
    assert payload["summary"]["profiler_status"] == "success"
    assert payload["summary"]["artifact_count"] == 2
    assert payload["summary"]["artifact_coverage_status"] == "complete"
    assert payload["summary"]["reason_codes"] == ["rocprof_artifacts_registered"]
    assert payload["summary"]["kernel_metrics"] == [
        {
            "kernel_name": "trace_counters",
            "name": "SQ_INSTS_VALU",
            "value": 12,
            "unit": "count",
            "source": "trace_counters.csv",
            "artifact": "trace_counters.csv",
            "parse_status": "available",
        }
    ]
    assert payload["summary"]["bottleneck_hints"][0]["category"] == "compute_bound"
    citation_kinds = {citation["kind"] for citation in payload["artifact_citations"]}
    assert citation_kinds == {"trace", "profile_metadata", "profiler_artifact"}
    profiler_citations = [
        citation
        for citation in payload["artifact_citations"]
        if citation["kind"] == "profiler_artifact"
    ]
    assert {citation["path"] for citation in profiler_citations} == {
        "trace.rocpd",
        "trace_counters.csv",
    }
    assert {citation["size_bytes"] for citation in profiler_citations} == {
        profile_artifact.stat().st_size,
        counter_artifact.stat().st_size,
    }
    for citation in payload["artifact_citations"]:
        assert "/" not in citation["path"]
        assert citation["sha256"] is not None
        assert len(citation["sha256"]) == 64
    assert "profile artifact" not in json.dumps(payload)


def test_profile_output_directory_tracks_trace_output(tmp_path: Path):
    output = tmp_path / "run" / "trace.jsonl"

    assert cli_sidecars._profile_output_directory(output, tmp_path) == (
        tmp_path / "run" / "trace.jsonl.rocprofv3"
    )


def test_static_evidence_paths_track_trace_output(tmp_path: Path):
    output = tmp_path / "run" / "trace.jsonl"
    staging = tmp_path / "staging"

    assert cli_sidecars._static_evidence_directory(output, staging) == (
        tmp_path / "run" / "trace.jsonl.static-evidence"
    )
    assert cli_sidecars._static_evidence_sidecar_path(output, staging) == (
        tmp_path / "run" / "trace.jsonl.static-evidence.json"
    )


def test_static_evidence_paths_fall_back_to_staging(tmp_path: Path):
    staging = tmp_path / "staging"

    assert cli_sidecars._static_evidence_directory(None, staging) == (
        staging / "static-evidence"
    )
    assert cli_sidecars._static_evidence_sidecar_path(None, staging) == (
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

    written = cli_sidecars._write_static_evidence_sidecar(output, staging, sidecar)

    assert written == tmp_path / "trace.jsonl.static-evidence.json"
    assert written is not None
    payload = json.loads(written.read_text())
    assert payload["schema_version"] == "sol_execbench.static_kernel_evidence.v1"
    assert payload["summary"]["status"] == "collected"
    assert payload["summary"]["artifact_count"] == 1
    assert payload["summary"]["claim_boundaries"]["diagnostic_only"] is True
    assert payload["summary"]["claim_boundaries"]["score_authority"] is False


def test_agent_feedback_sidecar_tracks_trace_output(tmp_path: Path):
    output = tmp_path / "trace.jsonl"

    assert cli_sidecars._agent_feedback_sidecar_path(output) == (
        tmp_path / "trace.jsonl.agent-feedback.json"
    )
    assert cli_sidecars._agent_feedback_sidecar_path(None) is None


def test_agent_feedback_sidecar_records_bounded_metadata(tmp_path: Path):
    output = tmp_path / "trace.jsonl"
    output.write_text('{"definition":"toy"}\n')
    solution = _solution()
    trace = Trace(
        definition="toy",
        solution="candidate",
        workload=Workload(
            uuid="w0",
            axes={"n": 1},
            inputs={"n": ScalarInput(value=1)},
        ),
        evaluation=Evaluation(
            status=EvaluationStatus.COMPILE_ERROR,
            environment=Environment(hardware="AMD gfx1200", libs={"hip": "7.0"}),
            timestamp="2026-06-16T00:00:00Z",
        ),
    )

    written = cli_sidecars._write_agent_feedback_sidecar(
        output,
        [trace],
        solution=solution,
        profile_result=None,
        static_evidence=None,
        run_id="run-001",
        feedback_target_id="gemm",
        feedback_candidate_id="candidate-sha",
        feedback_source_sha256="source-sha",
        feedback_sol_version="custom-sol-tag",
    )

    assert written == tmp_path / "trace.jsonl.agent-feedback.json"
    assert written is not None
    payload = json.loads(written.read_text())
    assert payload["schema_version"] == "sol_execbench.agent_feedback.v2"
    assert payload["authority"] == "diagnostic"
    assert payload["identity"]["trace_path"] == "trace.jsonl"
    assert payload["identity"]["target_id"] == "gemm"
    assert payload["identity"]["run_id"] == "run-001"
    assert payload["identity"]["candidate_id"] == "candidate-sha"
    assert payload["identity"]["source_sha256"] == "source-sha"
    assert payload["identity"]["sol_version"] == "custom-sol-tag"
    assert "candidate_hash" not in payload["identity"]
    assert "source_hash" not in payload["identity"]
    assert "sol_contract_version" not in payload["identity"]
    assert payload["summary"]["status_counts"] == {"COMPILE_ERROR": 1}
    assert payload["items"][0]["code"] == "compile_error"
    trace_citations = [
        citation
        for citation in payload["artifact_citations"]
        if citation["kind"] == "trace"
    ]
    assert trace_citations == [
        {
            "kind": "trace",
            "label": "canonical_trace_jsonl",
            "path": "trace.jsonl",
            "sha256": trace_citations[0]["sha256"],
            "status": None,
        }
    ]
    assert trace_citations[0]["sha256"] is not None
    assert len(trace_citations[0]["sha256"]) == 64
    assert "raw" not in json.dumps(payload).lower()


def test_agent_feedback_identity_uses_solution_source_hash(tmp_path: Path):
    output = tmp_path / "trace.jsonl"
    output.write_text('{"definition":"toy"}\n')
    trace = Trace(
        definition="toy",
        solution="candidate",
        workload=Workload(
            uuid="w0",
            axes={"n": 1},
            inputs={"n": ScalarInput(value=1)},
        ),
        evaluation=Evaluation(
            status=EvaluationStatus.COMPILE_ERROR,
            environment=Environment(hardware="AMD gfx1200", libs={"hip": "7.0"}),
            timestamp="2026-06-16T00:00:00Z",
        ),
    )
    first = _solution("def run(x):\n    return x\n")
    second = _solution("def run(x):\n    return x + 1\n")

    first_identity = cli_sidecars._agent_feedback_identity_fields(
        output,
        [trace],
        solution=first,
    )
    second_identity = cli_sidecars._agent_feedback_identity_fields(
        output,
        [trace],
        solution=second,
    )
    no_solution_identity = cli_sidecars._agent_feedback_identity_fields(output, [trace])

    assert first_identity["candidate_id"] == second_identity["candidate_id"]
    assert first_identity["source_sha256"] == first.hash()
    assert second_identity["source_sha256"] == second.hash()
    assert first_identity["source_sha256"] != second_identity["source_sha256"]
    assert no_solution_identity["source_sha256"] is None
    assert "candidate_hash" not in first_identity
    assert "source_hash" not in first_identity
    assert "candidate_hash" not in second_identity
    assert "source_hash" not in second_identity
    assert "candidate_hash" not in no_solution_identity
    assert "source_hash" not in no_solution_identity


def test_agent_feedback_identity_accepts_consumer_identity_fields(tmp_path: Path):
    output = tmp_path / "trace.jsonl"
    output.write_text('{"definition":"toy"}\n')
    trace = Trace(
        definition="toy",
        solution="candidate",
        workload=Workload(
            uuid="w0",
            axes={"n": 1},
            inputs={"n": ScalarInput(value=1)},
        ),
        evaluation=Evaluation(
            status=EvaluationStatus.COMPILE_ERROR,
            environment=Environment(hardware="AMD gfx1200", libs={"hip": "7.0"}),
            timestamp="2026-06-16T00:00:00Z",
        ),
    )

    identity = cli_sidecars._agent_feedback_identity_fields(
        output,
        [trace],
        solution=_solution(),
        run_id="run-001",
        target_id="gemm",
        candidate_id="candidate-sha",
        source_sha256="source-sha",
    )

    assert identity == {
        "target_id": "gemm",
        "run_id": "run-001",
        "candidate_id": "candidate-sha",
        "source_sha256": "source-sha",
    }


def test_static_evidence_none_does_not_collect(tmp_path: Path):
    sidecar = cli_sidecars._collect_static_evidence_for_cli(
        enabled=cli_sidecars.STATIC_EVIDENCE_NONE,
        is_cpp=True,
        staging_dir=tmp_path / "staging",
        output_file=tmp_path / "trace.jsonl",
    )

    assert sidecar is None


def test_static_evidence_auto_for_non_cpp_is_unsupported_sidecar(tmp_path: Path):
    sidecar = cli_sidecars._collect_static_evidence_for_cli(
        enabled=cli_sidecars.STATIC_EVIDENCE_AUTO,
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

    sidecar = cli_sidecars._collect_static_evidence_for_cli(
        enabled=cli_sidecars.STATIC_EVIDENCE_AUTO,
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

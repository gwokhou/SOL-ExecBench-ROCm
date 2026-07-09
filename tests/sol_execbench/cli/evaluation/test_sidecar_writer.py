from __future__ import annotations

import json
from pathlib import Path

from sol_execbench.cli.evaluation.sidecar_writer import write_optional_sidecars
from sol_execbench.core.bench.rocm_profiler import (
    Rocprofv3ProfileArtifact,
    Rocprofv3ProfileResult,
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
from sol_execbench.core.data.workload import ScalarInput, Workload


def test_optional_sidecars_use_consumer_identity_for_profile_summary(
    tmp_path: Path,
) -> None:
    output = tmp_path / "trace.jsonl"
    output.write_text('{"definition":"toy"}\n')

    write_optional_sidecars(
        output_file=output,
        staging_dir=tmp_path,
        traces=[_trace()],
        solution=_solution(),
        profile_result=_profile_result(tmp_path),
        static_evidence_result=None,
        trace_run_id="trace-sha-run-id",
        feedback_run_id="consumer-run-id",
        feedback_target_id="gemm",
        feedback_candidate_id="candidate-sha",
        feedback_source_sha256="source-sha",
        feedback_sol_version="consumer-sol-version",
    )

    profile_summary = json.loads(
        (tmp_path / "trace.jsonl.profile-summary.json").read_text()
    )
    agent_feedback = json.loads(
        (tmp_path / "trace.jsonl.agent-feedback.json").read_text()
    )

    assert profile_summary["identity"]["run_id"] == "consumer-run-id"
    assert profile_summary["identity"]["sol_version"] == "consumer-sol-version"
    assert profile_summary["identity"]["trace_path"] == "trace.jsonl"
    assert agent_feedback["identity"]["run_id"] == "consumer-run-id"
    assert agent_feedback["identity"]["sol_version"] == "consumer-sol-version"


def _profile_result(tmp_path: Path) -> Rocprofv3ProfileResult:
    artifact = tmp_path / "profile.rocpd"
    artifact.write_text("profile artifact\n")
    return Rocprofv3ProfileResult(
        status="success",
        command=("rocprofv3", "--kernel-trace", "--", "python", "eval_driver.py"),
        output_directory=tmp_path,
        output_file="profile",
        artifacts=(
            Rocprofv3ProfileArtifact(
                path=artifact,
                kind="rocpd",
                size_bytes=artifact.stat().st_size,
            ),
        ),
        returncode=0,
        profiler_available=True,
        artifact_coverage_status="complete",
        reason_codes=("rocprof_artifacts_registered",),
        timeout_seconds=60,
    )


def _solution() -> Solution:
    return Solution(
        name="candidate",
        definition="toy",
        author="agent",
        spec=BuildSpec(
            languages=[SupportedLanguages.PYTORCH],
            target_hardware=[SupportedHardware.LOCAL],
            entry_point="solution.py::run",
        ),
        sources=[SourceFile(path="solution.py", content="def run(x):\n    return x\n")],
    )


def _trace() -> Trace:
    return Trace(
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

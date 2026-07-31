# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Optional sidecar writing for the root evaluation workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sol_execbench.cli.commands import environment as cli_environment
from sol_execbench.cli.evaluation.compilation import CompilePhaseResult
from sol_execbench.cli.sidecars import (
    agent_feedback as cli_agent_feedback_sidecar,
    decision as cli_decision_sidecar,
    performance as cli_performance_sidecars,
    profile as cli_profile_sidecars,
    static_evidence as cli_static_evidence,
)
from sol_execbench.core.bench.rocm_profiler import Rocprofv3ProfileResult
from sol_execbench.core.bench.static_kernel.evidence import (
    StaticKernelEvidenceSidecar,
)
from sol_execbench.core.data.solution import Solution
from sol_execbench.core.data.trace import Trace


@dataclass(frozen=True, slots=True, kw_only=True)
class SidecarIdentity:
    """Consumer-supplied identity shared by optional diagnostic sidecars."""

    trace_run_id: str | None
    feedback_run_id: str | None
    target_id: str | None
    candidate_id: str | None
    source_sha256: str | None
    sol_version: str | None

    @property
    def run_id(self) -> str | None:
        """Prefer an explicit consumer run id over the canonical trace id."""
        return self.feedback_run_id or self.trace_run_id


@dataclass(frozen=True, slots=True, kw_only=True)
class SidecarWriteRequest:
    """Typed inputs for writing every optional evaluation sidecar."""

    output_file: Path | None
    staging_dir: Path
    traces: list[Trace]
    solution: Solution
    profile_result: Rocprofv3ProfileResult | None
    static_evidence_result: StaticKernelEvidenceSidecar | None
    decision: str
    identity: SidecarIdentity
    compile_result: CompilePhaseResult | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class WrittenSidecars:
    """Paths successfully written during optional sidecar publication."""

    environment: Path | None
    profile: Path | None
    profile_summary: Path | None
    static_evidence: Path | None
    decision: Path | None
    agent_feedback: Path | None
    performance_timing: Path | None
    performance_access: Path | None
    performance_evidence: Path | None
    performance_replay: Path | None


@dataclass(frozen=True, slots=True)
class _PerformanceSidecars:
    timing: Path | None
    access: Path | None
    replay: Path | None
    evidence: Path | None


def write_optional_sidecars(request: SidecarWriteRequest) -> WrittenSidecars:
    """Write optional sidecars from one immutable request."""
    environment_sidecar_path = (
        cli_environment._write_environment_snapshot_sidecar(
            request.output_file,
        )
    )
    profile_sidecar_path = cli_profile_sidecars._write_profile_sidecar(
        request.output_file,
        request.profile_result,
    )
    profile_summary_sidecar_path = (
        cli_profile_sidecars._write_profile_summary_sidecar(
            request.output_file,
            request.profile_result,
            profile_sidecar_path=profile_sidecar_path,
            run_id=request.identity.run_id,
            sol_version=request.identity.sol_version,
        )
    )
    static_evidence_sidecar_path = (
        cli_static_evidence._write_static_evidence_sidecar(
            request.output_file,
            request.staging_dir,
            request.static_evidence_result,
        )
    )
    performance = _write_performance_sidecars(
        request,
        profile_summary_path=profile_summary_sidecar_path,
        static_evidence_path=static_evidence_sidecar_path,
    )
    decision_sidecar_path, agent_feedback_sidecar_path = (
        _write_decision_and_feedback(
            request,
            environment_sidecar_path=environment_sidecar_path,
            profile_sidecar_path=profile_sidecar_path,
            static_evidence_sidecar_path=static_evidence_sidecar_path,
        )
    )
    return WrittenSidecars(
        environment=environment_sidecar_path,
        profile=profile_sidecar_path,
        profile_summary=profile_summary_sidecar_path,
        static_evidence=static_evidence_sidecar_path,
        decision=decision_sidecar_path,
        agent_feedback=agent_feedback_sidecar_path,
        performance_timing=performance.timing,
        performance_access=performance.access,
        performance_evidence=performance.evidence,
        performance_replay=performance.replay,
    )


def _write_performance_sidecars(
    request: SidecarWriteRequest,
    *,
    profile_summary_path: Path | None,
    static_evidence_path: Path | None,
) -> _PerformanceSidecars:
    timing = cli_performance_sidecars.write_performance_timing_sidecar(
        output_file=request.output_file,
        staging_dir=request.staging_dir,
        traces=request.traces,
        solution=request.solution,
    )
    access = cli_performance_sidecars.write_performance_access_sidecar(
        output_file=request.output_file,
        staging_dir=request.staging_dir,
        traces=request.traces,
    )
    replay = cli_performance_sidecars.write_performance_replay_sidecar(
        output_file=request.output_file,
        staging_dir=request.staging_dir,
        traces=request.traces,
        solution=request.solution,
        timing_path=timing,
        profile_result=request.profile_result,
        static_evidence=request.static_evidence_result,
        compile_result=request.compile_result,
    )
    evidence = cli_performance_sidecars.write_performance_evidence_manifest(
        cli_performance_sidecars.PerformanceEvidenceManifestWriteRequest(
            output_file=request.output_file,
            traces=request.traces,
            solution=request.solution,
            timing_path=timing,
            access_path=access,
            profile_summary_path=profile_summary_path,
            static_evidence_path=static_evidence_path,
            profile_result=request.profile_result,
            static_evidence=request.static_evidence_result,
            compile_result=request.compile_result,
            replay_path=replay,
        )
    )
    return _PerformanceSidecars(
        timing=timing,
        access=access,
        replay=replay,
        evidence=evidence,
    )


def _write_decision_and_feedback(
    request: SidecarWriteRequest,
    *,
    environment_sidecar_path: Path | None,
    profile_sidecar_path: Path | None,
    static_evidence_sidecar_path: Path | None,
) -> tuple[Path | None, Path | None]:
    decision = cli_decision_sidecar._write_decision_sidecar(
        request.output_file,
        request.decision,
        request.static_evidence_result,
        environment_sidecar_path,
        profile_result=request.profile_result,
        run_id=request.identity.run_id,
        target_id=request.identity.target_id,
        candidate_id=request.identity.candidate_id,
        source_sha256=request.identity.source_sha256,
        sol_version=request.identity.sol_version,
    )
    feedback = cli_agent_feedback_sidecar._write_agent_feedback_sidecar(
        cli_agent_feedback_sidecar.AgentFeedbackWriteRequest(
            output_file=request.output_file,
            traces=request.traces,
            solution=request.solution,
            profile_result=request.profile_result,
            static_evidence=request.static_evidence_result,
            identity=cli_agent_feedback_sidecar.AgentFeedbackIdentityOverrides(
                run_id=request.identity.run_id,
                target_id=request.identity.target_id,
                candidate_id=request.identity.candidate_id,
                source_sha256=request.identity.source_sha256,
                sol_version=request.identity.sol_version,
            ),
            artifact_paths=cli_agent_feedback_sidecar.AgentFeedbackArtifactPaths(
                environment=environment_sidecar_path,
                profile=profile_sidecar_path,
                static_evidence=static_evidence_sidecar_path,
            ),
        ),
    )
    return decision, feedback


__all__ = [
    "SidecarIdentity",
    "SidecarWriteRequest",
    "WrittenSidecars",
    "write_optional_sidecars",
]

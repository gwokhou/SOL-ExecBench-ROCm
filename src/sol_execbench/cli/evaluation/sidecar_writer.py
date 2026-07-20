# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Optional sidecar writing for the root evaluation workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sol_execbench.core.bench.rocm_profiler import Rocprofv3ProfileResult
from sol_execbench.core.bench.static_kernel.evidence import StaticKernelEvidenceSidecar
from sol_execbench.core.data.solution import Solution
from sol_execbench.core.data.trace import Trace

from ..commands import environment as cli_environment
from ..sidecars import agent_feedback as cli_agent_feedback_sidecar
from ..sidecars import decision as cli_decision_sidecar
from ..sidecars import profile as cli_profile_sidecars
from ..sidecars import static_evidence as cli_static_evidence


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


@dataclass(frozen=True, slots=True, kw_only=True)
class WrittenSidecars:
    """Paths successfully written during optional sidecar publication."""

    environment: Path | None
    profile: Path | None
    profile_summary: Path | None
    static_evidence: Path | None
    decision: Path | None
    agent_feedback: Path | None


def write_optional_sidecars(request: SidecarWriteRequest) -> WrittenSidecars:
    """Write optional sidecars from one immutable request."""
    environment_sidecar_path = cli_environment._write_environment_snapshot_sidecar(
        request.output_file
    )
    profile_sidecar_path = cli_profile_sidecars._write_profile_sidecar(
        request.output_file, request.profile_result
    )
    profile_summary_sidecar_path = cli_profile_sidecars._write_profile_summary_sidecar(
        request.output_file,
        request.profile_result,
        profile_sidecar_path=profile_sidecar_path,
        run_id=request.identity.run_id,
        sol_version=request.identity.sol_version,
    )
    static_evidence_sidecar_path = cli_static_evidence._write_static_evidence_sidecar(
        request.output_file,
        request.staging_dir,
        request.static_evidence_result,
    )
    decision_sidecar_path = cli_decision_sidecar._write_decision_sidecar(
        request.output_file,
        request.decision,
        request.static_evidence_result,
        environment_sidecar_path,
        runtime_profile_available=profile_summary_sidecar_path is not None,
        run_id=request.identity.run_id,
        target_id=request.identity.target_id,
        candidate_id=request.identity.candidate_id,
        source_sha256=request.identity.source_sha256,
        sol_version=request.identity.sol_version,
    )
    agent_feedback_sidecar_path = (
        cli_agent_feedback_sidecar._write_agent_feedback_sidecar(
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
            )
        )
    )
    return WrittenSidecars(
        environment=environment_sidecar_path,
        profile=profile_sidecar_path,
        profile_summary=profile_summary_sidecar_path,
        static_evidence=static_evidence_sidecar_path,
        decision=decision_sidecar_path,
        agent_feedback=agent_feedback_sidecar_path,
    )


__all__ = [
    "SidecarIdentity",
    "SidecarWriteRequest",
    "WrittenSidecars",
    "write_optional_sidecars",
]

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 contributors to SOL ExecBench ROCm Port
# SPDX-License-Identifier: Apache-2.0

"""Agent feedback sidecar helpers for the SOL-ExecBench CLI."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from rich.console import Console

from ...core.bench.agent_feedback import (
    AgentFeedbackArtifactCitation,
    artifact_citation_from_path,
    build_agent_feedback_sidecar,
)
from ...core.bench.rocm_profiler import Rocprofv3ProfileResult
from ...core.bench.static_kernel.evidence import StaticKernelEvidenceSidecar
from ...core.evidence.checksums import sha256_file, stable_json_checksum
from ...core.data.solution import Solution
from ...core.data.trace import Trace
from ...core.evidence.runtime_evidence import write_json_payload

console = Console(stderr=True)


def _agent_feedback_sidecar_path(output_file: Path | None) -> Path | None:
    """Return the optional agent feedback sidecar path for a trace output."""

    if output_file is None:
        return None
    return output_file.with_name(f"{output_file.name}.agent-feedback.json")


def _write_agent_feedback_sidecar(
    output_file: Path | None,
    traces: list[Trace],
    *,
    solution: Solution | None = None,
    profile_result: Rocprofv3ProfileResult | None,
    static_evidence: StaticKernelEvidenceSidecar | None,
    environment_sidecar_path: Path | None = None,
    profile_sidecar_path: Path | None = None,
    static_evidence_sidecar_path: Path | None = None,
    run_id: str | None = None,
    feedback_target_id: str | None = None,
    feedback_candidate_id: str | None = None,
    feedback_source_sha256: str | None = None,
    feedback_sol_version: str | None = None,
) -> Path | None:
    """Write optional agent feedback metadata without changing trace JSONL."""

    sidecar_path = _agent_feedback_sidecar_path(output_file)
    if sidecar_path is None:
        return None

    try:
        if run_id is None:
            run_id = _agent_feedback_run_id(output_file, traces)
        identity_fields = _agent_feedback_identity_fields(
            output_file,
            traces,
            solution=solution,
            run_id=run_id,
            target_id=feedback_target_id,
            candidate_id=feedback_candidate_id,
            source_sha256=feedback_source_sha256,
        )
        sidecar = build_agent_feedback_sidecar(
            traces=traces,
            profile_result=profile_result,
            static_evidence=static_evidence,
            trace_path=str(output_file) if output_file is not None else None,
            target_id=identity_fields["target_id"],
            run_id=identity_fields["run_id"],
            candidate_id=identity_fields["candidate_id"],
            source_sha256=identity_fields["source_sha256"],
            sol_version=feedback_sol_version,
            artifact_citations=_agent_feedback_artifact_citations(
                output_file=output_file,
                environment_sidecar_path=environment_sidecar_path,
                profile_sidecar_path=profile_sidecar_path,
                static_evidence_sidecar_path=static_evidence_sidecar_path,
            ),
        )
        write_json_payload(sidecar_path, sidecar)
        console.print(f"[green]Saved agent feedback to {sidecar_path}[/green]")
        return sidecar_path
    except Exception as exc:
        console.print(f"[yellow]Agent feedback metadata skipped: {exc}[/yellow]")
        return None


def _agent_feedback_identity_fields(
    output_file: Path | None,
    traces: Sequence[Trace],
    *,
    solution: Solution | None = None,
    run_id: str | None = None,
    target_id: str | None = None,
    candidate_id: str | None = None,
    source_sha256: str | None = None,
) -> dict[str, str | None]:
    """Derive stable feedback freshness identity from emitted trace data."""

    target_records = [
        {
            "definition": trace.definition,
            "workload_uuid": trace.workload.uuid,
        }
        for trace in traces
    ]
    solution_labels = sorted(
        {trace.solution for trace in traces if trace.solution is not None}
    )

    return {
        "target_id": (
            target_id
            if target_id is not None
            else (stable_json_checksum(target_records) if target_records else None)
        ),
        "run_id": (
            run_id
            if run_id is not None
            else _agent_feedback_run_id(output_file, traces)
        ),
        "candidate_id": (
            candidate_id
            if candidate_id is not None
            else (stable_json_checksum(solution_labels) if solution_labels else None)
        ),
        "source_sha256": (
            source_sha256
            if source_sha256 is not None
            else (solution.hash() if solution is not None else None)
        ),
    }


def _agent_feedback_run_id(
    output_file: Path | None,
    traces: Sequence[Trace],
) -> str | None:
    """Return a stable run id from the persisted trace file or trace payload."""

    if output_file is not None and output_file.is_file():
        return sha256_file(output_file)
    if not traces:
        return None
    return stable_json_checksum([trace.model_dump(mode="json") for trace in traces])


def _agent_feedback_artifact_citations(
    *,
    output_file: Path | None,
    environment_sidecar_path: Path | None,
    profile_sidecar_path: Path | None,
    static_evidence_sidecar_path: Path | None,
    trace_sha256: str | None = None,
) -> list[AgentFeedbackArtifactCitation]:
    """Return compact citations for artifacts written during this CLI run."""

    citations: list[AgentFeedbackArtifactCitation] = []
    if output_file is not None:
        citations.append(
            artifact_citation_from_path(
                kind="trace",
                label="canonical_trace_jsonl",
                path=output_file,
                sha256=trace_sha256,
            )
        )
    for kind, label, path in (
        ("environment", "environment_snapshot", environment_sidecar_path),
        ("profile", "rocprofv3_profile", profile_sidecar_path),
        ("static_evidence", "static_kernel_evidence", static_evidence_sidecar_path),
    ):
        if path is not None:
            citations.append(
                artifact_citation_from_path(kind=kind, label=label, path=path)
            )
    return citations
